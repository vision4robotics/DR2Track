function [g_f,g_pre] = run_training(xf,use_sz,g_pre,params,mu,yf,w)
        g_f = single(zeros(size(xf)));
        h_f = g_f;
        l_f = h_f;
        gamma = 1;
        betha = 10;
        gamma_max = 10000;
        % ADMM solution    
        T = prod(use_sz);
        S_xx = sum(conj(xf) .* xf, 3);
        Sg_pre= sum(conj(xf) .* g_pre, 3);
        Sgx_pre= bsxfun(@times, xf, Sg_pre);
        iter = 1;
        while (iter <= params.admm_iterations)
            % subproblem g
            B = S_xx + T * (gamma + mu);
            Shx_f = sum(conj(xf) .* h_f, 3);
            Slx_f = sum(conj(xf) .* l_f, 3);
            g_f = ((1/(T*(gamma + mu)) * bsxfun(@times,  yf, xf)) - ((1/(gamma + mu)) * l_f) +(gamma/(gamma + mu)) * h_f) + (mu/(gamma + mu)) * g_pre - ...
                bsxfun(@rdivide,(1/(T*(gamma + mu)) * bsxfun(@times, xf, (S_xx .*  yf)) + (mu/(gamma + mu)) * Sgx_pre- ...
                (1/(gamma + mu))* (bsxfun(@times, xf, Slx_f)) +(gamma/(gamma + mu))* (bsxfun(@times, xf, Shx_f))), B);
            %   subproblem h
            lhd= T ./  (params.admm_lambda*w .^2 + gamma*T); 
            X=ifft2(gamma*(g_f + l_f));
            h=bsxfun(@times,lhd,X);
            h_f = fft2(h);
            %   update h
            l_f = l_f + (gamma * (g_f - h_f));
            %   update gamma
            gamma = min(betha* gamma, gamma_max);
            iter = iter+1;
        end
        % save the trained filters
        g_pre= g_f;
     
end

