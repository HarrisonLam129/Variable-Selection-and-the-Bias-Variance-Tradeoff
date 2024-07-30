function [betaCV] = crossval(T, sparse)
    dim = size(T);
    N = dim(1);
    p = dim(2)-1;
    y = T(:, 1);
    x = T(:, 2:dim(2));
    
    %Generating random permutation and permuting data by row 
    perm = randperm(N);
    y = y(perm, :);
    x = x(perm, :);
    PE = zeros(1, p);

    B = sparse(T)
    for j = 1:p
            prevEnd = 0;
            sum = 0;
        for k = 1:10
            %Splitting T into T^k and T^-k
            testN = prevEnd+1:(N*k/10);
            testY = y(testN, :);
            testX = x(testN, :);
            trainN = setdiff(1:N, testN);
            trainY = y(trainN, :);
            trainX = x(trainN, :);
            prevEnd = floor(N*k/10);
            %Finding beta for training data T^-k
            betaMatrix = sparse([trainY trainX]);
            beta = betaMatrix(:, j);
            %Calculating RSS
            RSS = (1/N)*(norm(testY-testX*beta))^2;
            sum = sum + RSS;
        end
        PE(j) = sum/10;
    end
    %Finding and returning candidate with lowest PE
    PE
    [~, minIndex] = min(PE);
    betaCV = B(:, minIndex);
end