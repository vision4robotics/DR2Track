% This function implements the ASRCF tracker.

function [results] = tracker(params)

num_frames     = params.no_fram;
newton_iterations=params.newton_iterations;
global_feat_params = params.t_global;
featureRatio = params.t_global.cell_size;
search_area = prod(params.wsize * params.search_area_scale);
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);

[currentScaleFactor,base_target_sz,reg_sz,sz,use_sz] = init_size(params,target_sz,search_area);
[y_0,cos_window] = init_gauss_win(params,base_target_sz,featureRatio,use_sz);
[features,im,colorImage] = init_features(params);
[ysf,scale_window,scaleFactors,scale_model_sz,min_scale_factor,max_scale_factor] = init_scale(params,target_sz,sz,base_target_sz,im);
% Pre-computes the grid that is used for score optimization
ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
time = 0;
loop_frame = 1;
Vy=0;
Vx=0;
avg_list=zeros(num_frames,1);
avg_list(1)=0;

for frame = 1:num_frames
    im = load_image(params,frame,colorImage);
    tic();  
%% main loop

    if frame > 1
        pos_pre=pos;
        [xtf,pos,translation_vec,response,disp_row,disp_col] = run_detection(im,pos,sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame);
        Vy=pos(1)-pos_pre(1);
        Vx=pos(2)-pos_pre(2);
        % search for the scale of object
        [xs,currentScaleFactor,recovered_scale]  = search_scale(sf_num,sf_den,im,pos,base_target_sz,currentScaleFactor,scaleFactors,scale_window,scale_model_sz,min_scale_factor,max_scale_factor,params);
    end
    % update the target_sz via currentScaleFactor
    target_sz =round(base_target_sz * currentScaleFactor);
    %save position
     rect_position(loop_frame,:) =[pos([2,1]) - (target_sz([2,1]))/2, target_sz([2,1])];
     if frame==1 
            % extract training sample image region
             pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
             pixels = uint8(gather(pixels));
             x=get_features(pixels,features,params.t_global);
             xf=fft2(bsxfun(@times,x,cos_window));
     else
           % use detection features
            shift_samp_pos = 2*pi * translation_vec ./(currentScaleFactor* sz);
            xf = shift_sample(xtf, shift_samp_pos, kx', ky');
     end
        if  frame == 1
            [~,~,w]=init_regwindow(use_sz,reg_sz,params);
            g_pre= zeros(size(xf));
            mu = 0;
        else
            mu=params.init_mu;
        end
        
         if frame>1
            peak=max(response(:));
            response=response/max(response(:));
            response=circshift(response,floor([size(response,1),size(response,2)]/2));
            response=circshift(response,round([-disp_row,-disp_col]));
            bg_response=bsxfun(@times,response,w/1e5);
            y=circshift(y_0,floor(([size(y_0,1),size(y_0,2)]/2)));
            BW = imregionalmax(response);
            Bys=floor(size(BW,1)/2-reg_sz(1)/2):floor(size(BW,1)/2+reg_sz(1)/2);
            Bxs=floor(size(BW,2)/2-reg_sz(2)/2):floor(size(BW,2)/2+reg_sz(2)/2);
            BW(Bys,Bxs)=0;
            CC = bwconncomp(BW);
            local_max = [max(response(:)) 0];
            if length(CC.PixelIdxList) > 1
                local_max = zeros(length(CC.PixelIdxList),1);
                for i = 1:length(CC.PixelIdxList)
                    local_max(i) = response(CC.PixelIdxList{i}(1));
                end
                local_max = sort(local_max, 'descend');
            end
            if length(local_max)<params.local_nums
                num_max=length(local_max);
            else
                num_max=params.local_nums;
            end
            sum_local=0;
            for i=1:num_max
                [row,col]=find(bg_response==local_max(i));
                y(row,col)=-params.beta*local_max(i);
                sum_local=sum_local+local_max(i);
            end
            avg_local=sum_local/num_max*peak;
            avg_list(frame)=avg_local;
            y=circshift(y,-floor(([size(y,1),size(y,2)]/2)));
            yf=fft2(y);
         else
            y=y_0;
            yf=fft2(y);
         end
    
         [g_f,g_pre] = run_training(xf,use_sz,g_pre,params,mu,yf,w);
            
     
        %% Update Scale
        if frame==1
            xs = crop_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
        else
            xs= shift_sample_scale(im, pos, base_target_sz,xs,recovered_scale,currentScaleFactor*scaleFactors,scale_window,scale_model_sz);
        end
        xsf = fft(xs,[],2);
        new_sf_num = bsxfun(@times, ysf, conj(xsf));
        new_sf_den = sum(xsf .* conj(xsf), 1);
        if frame == 1
            sf_den = new_sf_den;
            sf_num = new_sf_num;
        else
            sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
            sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
        end

     time = time + toc();

     %%   visualization
     if params.visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        figure(1);
        imshow(im);
        if frame == 1
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(12, 26, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            hold off;
        else
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(12, 28, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            text(12, 66, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            hold off;
         end
        drawnow
    end
     loop_frame = loop_frame + 1;
end

%   save resutls.
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
end
