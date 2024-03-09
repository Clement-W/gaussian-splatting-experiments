#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
import torch.nn.functional as F
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import compute_sobel, psnr, compute_laplacian
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)


        if(gaussians.active_sh_degree < gaussians.max_sh_degree and dataset.start_full_sh == 1):
            gaussians.active_sh_degree = gaussians.max_sh_degree
        # Every 1000 its we increase the levels of SH up to a maximum degree
        elif lp.start_full_sh == 0 and iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera until the list is empty then repopulate it
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # viewspace_point_tensor.shape
        # torch.Size([66290, 3])
        # len(radii) = 66290
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # visibility filter to get true/false for the gaussians we can observe from the viewpoint cam
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        if opt.loss_type == "l1_ssim":
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        elif(opt.loss_type == "l1"):
            Ll1 = l1_loss(image, gt_image)
            loss = Ll1
        elif(opt.loss_type == "ssim"):
            loss = 1.0 - ssim(image, gt_image)
            Ll1 = loss # only used for logging, to avoid modifying the tensorboard logging function
        elif(opt.loss_type == "l2"):
            loss = l2_loss(image, gt_image)
            Ll1 = loss # only used for logging
        elif(opt.loss_type == "l2_ssim"):
            Ll2 = l2_loss(image, gt_image)
            Ll1 = Ll2 # only used for logging
            loss = (1.0 - opt.lambda_dssim) * Ll2 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        elif(opt.loss_type == "huber_ssim"):
            huber = F.smooth_l1_loss(image, gt_image)
            Ll1 = huber # only used for logging
            loss = (1.0 - opt.lambda_dssim) * huber + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            raise ValueError(f"Unknown loss type {opt.loss_type}")
        

        # if(opt.regularization_type != "" and iteration > opt.regularize_from_iter and iteration < opt.regularize_until_iter):
                        
        #     # if(opt.regularization_type == "variance_regularization"):
        #     #     norm_scaling = torch.norm(gaussians.get_scaling, dim=1) #essayer gaussians.get_scaling[visibility_filter]
        #     #     loss += opt.lambda_regularization * torch.mean(norm_scaling)
        #     if(opt.regularization_type == "maxvariance_regularization"):
        #         max_scaling = torch.max(gaussians.get_scaling, dim=1).values #essayer gaussians.get_scaling[visibility_filter]
        #         loss += opt.lambda_regularization * torch.mean(max_scaling)
        #     elif(opt.regularization_type == "opacity_regularization"):
        #         opacities = gaussians.get_opacity
        #         loss += opt.lambda_regularization * (-opacities * torch.log(opacities+1e-9) - (1 - opacities) * torch.log(1 - opacities+1e-9)).mean()
        #     elif(opt.regularization_type == "edge_regularization"):
        #         sobel_gt = viewpoint_cam.image_edges.cuda()
        #         sobel_render = compute_sobel(image)
        #         loss += opt.lambda_regularization * l1_loss(sobel_render, sobel_gt)
        #     elif(opt.regularization_type == "smoothness_regularization"):
        #         smoothness = torch.norm(compute_laplacian(image), p=2)
        #         loss += opt.lambda_regularization * smoothness
        #     else:
        #         raise ValueError(f"Unknown regularization type {opt.regularization_type}")


 
        # allows to add multiple regularizations at different times
        if(opt.maxvariance_regularization != 0 and iteration > 15_000 and iteration < 30_000):
            max_scaling = torch.max(gaussians.get_scaling, dim=1).values
            loss += opt.maxvariance_regularization * torch.mean(max_scaling)

        if(opt.opacity_regularization != 0 and iteration > 15_000 and iteration < 30_000):
            opacities = gaussians.get_opacity
            loss += opt.opacity_regularization * (-opacities * torch.log(opacities+1e-9) - (1 - opacities) * torch.log(1 - opacities+1e-9)).mean()

        if(opt.edge_regularization != 0 and iteration > 500 and iteration < 15_000):
            sobel_gt = viewpoint_cam.image_edges.cuda()
            sobel_render = compute_sobel(image)
            loss += opt.edge_regularization * l1_loss(sobel_render, sobel_gt)

        if(opt.smoothness_regularization != 0 and iteration > 500 and iteration < 30_000):
            smoothness = torch.norm(compute_laplacian(image), p=2)
            loss += opt.smoothness_regularization * smoothness

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # store informations about the gradients
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity() # we prune also right before the first time we dansify if white background

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # save final model information
    gaussians_count = scene.gaussians.get_xyz.shape[0]
    with open(scene.model_path + '/gaussiancount.txt', 'w') as f:
        f.write(str(gaussians_count))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('nb_gaussians', scene.gaussians.get_xyz.shape[0],iteration)

        if(iteration%50==0):
            grads_xyz=scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
            grads_xyz[grads_xyz.isnan()]=0.0

            grads_featuresdc=scene.gaussians.features_dc_gradient_accum / scene.gaussians.denom
            grads_featuresdc[grads_featuresdc.isnan()]=0.0

            grads_featuresrest=scene.gaussians.features_rest_gradient_accum / scene.gaussians.denom
            grads_featuresrest[grads_featuresrest.isnan()]=0.0

            grads_scaling=scene.gaussians.scaling_gradient_accum / scene.gaussians.denom
            grads_scaling[grads_scaling.isnan()]=0.0

            grads_rotation=scene.gaussians.rotation_gradient_accum / scene.gaussians.denom
            grads_rotation[grads_rotation.isnan()]=0.0

            grads_opacity=scene.gaussians.opacity_gradient_accum / scene.gaussians.denom
            grads_opacity[grads_opacity.isnan()]=0.0

            tb_writer.add_scalar('xyz_gradmean', grads_xyz.mean(),iteration)
            tb_writer.add_scalar('featuresdc_gradmean', grads_featuresdc.mean(),iteration)
            tb_writer.add_scalar('featuresrest_gradmean', grads_featuresrest.mean(),iteration)
            tb_writer.add_scalar('scaling_gradmean', grads_scaling.mean(),iteration)
            tb_writer.add_scalar('rotation_gradmean', grads_rotation.mean(),iteration)
            tb_writer.add_scalar('opacity_gradmean', grads_opacity.mean(),iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])     
                ssim_test /= len(config['cameras'])  

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[4_000, 7_000, 14_000, 21_000, 26_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
