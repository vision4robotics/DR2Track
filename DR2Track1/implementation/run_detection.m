function [xtf,pos,translation_vec,response,disp_row,disp_col] = run_detection(im,pos,sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame)
        center=pos+[Vy Vx];
        pixel_template=get_pixels(im, center, round(sz*currentScaleFactor), sz);             
        xt=get_features(pixel_template,features,global_feat_params);
        xtf=fft2(bsxfun(@times,xt,cos_window));    
        savedir='H:\IROS\Ablation\features\';
%         if frame==295
%         xt_f=ifft2(xtf,'symmetric');
%         Xt=sum(xt_f,3);
%         colormap(jet);
%         surf(Xt);
%         shading interp;
%         axis ij;
%         axis off;
%         view([34,50]);
%         saveas(gcf,[savedir,num2str(frame),'.png']);
%         end
        responsef=permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
        % if we undersampled features, we want to interpolate the
        % response so it has the same size as the image patch
        responsef_padded = resizeDFT2(responsef, use_sz);
        % response in the spatial domain
        response = ifft2(responsef_padded, 'symmetric');
        % find maximum peak
        [disp_row, disp_col] = resp_newton(response, responsef_padded,newton_iterations, ky, kx, use_sz);
        % calculate translation
        translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
        %update position
        pos = center + translation_vec;
end

