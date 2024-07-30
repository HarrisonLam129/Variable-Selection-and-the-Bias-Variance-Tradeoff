function [B] = greedysubset(T)
    dim = size(T);
    N = dim(1);
    p = dim(2)-1;
    B = zeros(p);
    y = T(:, 1);
    x = T(:, 2:dim(2));
    
    %Finding the first regressor with lowest RSS
    RSSArray = zeros(1, p);
    for j = 1:p
        reducedX = x(:, j);
        beta = (reducedX'*reducedX)\(reducedX'*y);
        RSSArray(j) = (1/N)*(norm(y-reducedX*beta))^2;
    end
    [~, minIndex] = min(RSSArray);
    currentModel = minIndex;
    reducedX = x(:, currentModel);
    beta = (reducedX'*reducedX)\(reducedX'*y);
    B(currentModel, 1) = beta;

    while length(currentModel) < p
        numReg = length(currentModel);
        RSSArray = zeros(1, p-numReg);
        %Finding all the models containing M_d, with one extra regressor
        greedyModels = setdiff(1:p, currentModel)';
        greedyModels = [greedyModels repmat(currentModel, [p-numReg 1])];
        greedyModels = sort(greedyModels, 2);
        %Looping through all the models in greedyModels
        for j = 1:p-numReg
            model = greedyModels(j,:);
            reducedX = x(:, model);
            beta = (reducedX'*reducedX)\(reducedX'*y);
            RSSArray(j) = (1/N)*(norm(y-reducedX*beta))^2;
        end
        %Displaying RSS for each added covariate
        disp([setdiff(1:p, currentModel); RSSArray]);
        %Finding the regressor that gives the smallest RSS
        [~, minIndex] = min(RSSArray);
        currentModel = greedyModels(minIndex,:);
        %Calculating beta again
        reducedX = x(:, currentModel);
        beta = (reducedX'*reducedX)\(reducedX'*y);
        %Storing the values in B
        for j = 1:length(currentModel)
            B(currentModel(j), length(currentModel)) = beta(j);
        end
    end
end