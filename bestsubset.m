function [B] = bestsubset(T)
    dim = size(T);
    N = dim(1);
    p = dim(2)-1;
    B = zeros(p);
    y = T(:, 1);
    x = T(:, 2:dim(2));
    for modelSize = 1:p
        %Finding the best subset of size modelSize
        models = nchoosek(1:p, modelSize);
        numModels = length(models(:,1));
        bestModel = models(1,:);
        %Looping through all the models of size modelSize
        for j = 1:numModels
            model = models(j,:);
            reducedX = x(:, model);
            beta = (reducedX'*reducedX)\(reducedX'*y);
            RSS = (1/N)*(norm(y-reducedX*beta))^2;
            if j == 1
                min = RSS;
                bestModelBeta = beta;
            else
                %Comparing RSS of current model to smallest RSS so far
                if RSS < min
                    min = RSS;
                    bestModel = models(j,:);
                    bestModelBeta = beta;
                end
            end
        end
        %Setting the corresponding values of B to values of beta^(M_j)
        for j = 1:modelSize
            B(bestModel(j), modelSize) = bestModelBeta(j);
        end
    end
end