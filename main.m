%Question 2
trainingSize = 200;
testSize = 1000;
p = 10;
mu = zeros(1, p);
sigmaSquared = 1;
beta = [-0.5 0.45 -0.4 0.35 -0.3 0.25 -0.2 0.15 -0.1 0.05]';

avgTrainingRSS = zeros(1, p);
avgTestRSS = zeros(1, p);

for i = 1:100
    %Generating training data
    trainingX = mvnrnd(mu, eye(p), trainingSize);
    trainingNoise = mvnrnd(zeros(1, trainingSize), sigmaSquared*eye(trainingSize), 1);
    trainingY = trainingX * beta + trainingNoise';
    %Generating test data
    testX = mvnrnd(mu, eye(p), testSize);
    testNoise = mvnrnd(zeros(1, testSize), sigmaSquared*eye(testSize), 1);
    testY = testX * beta + testNoise';
    %Calculating estimate
    betaEstimateMatrix = zeros(p);
    trainingRSS = zeros(1, p);
    testRSS = zeros(1, p);
    for j = 1:p
        reducedX = trainingX(:, 1:j);
        reducedBetaLS = (reducedX'*reducedX)\(reducedX'*trainingY);
        betaEstimateMatrix(j, :) = [reducedBetaLS' zeros(1, p-j)];
        trainingRSS(j) = (1/trainingSize)*(norm(trainingY-reducedX*reducedBetaLS))^2;
        testRSS(j) = (1/testSize)*(norm(testY-testX(:, 1:j)*reducedBetaLS))^2;
    end
    avgTrainingRSS = avgTrainingRSS + trainingRSS;
    avgTestRSS = avgTestRSS + testRSS;
end

avgTrainingRSS = avgTrainingRSS/100;
avgTestRSS = avgTestRSS/100;
plot(1:p, avgTestRSS, 1:p, avgTrainingRSS)
title(append('Training size = ', num2str(trainingSize), ', test size = ', num2str(testSize)))
legend('avgTestRSS', 'avgTrainingRSS')
xlabel('j')

B = bestsubset([trainingY trainingX]);
disp(B);
disp(reducedBetaLS);

B = greedysubset([trainingY trainingX]);
disp(B);

B = greedysubsetF([trainingY trainingX]);
disp(B);

betaCV = crossval([trainingY trainingX], @bestsubset)

pData = readtable('II-10-15-2022-prostate.dat');
pData.Properties.VariableNames = {'lpsa','lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45'};

newCovariates = mvnrnd(zeros(1,4), eye(4), 97);
newCovTable = array2table(newCovariates);
newData = [pData newCovTable];
rng('default')
perm = randperm(97);
newData = newData(perm, :);
trainingData = newData(1:70, :);
testData = newData(71:97, :);

bestRegressorMatrix = zeros(12, 4);
functions = {@bestsubset, @greedysubset, @greedysubsetF, @monotonic_lars};
for i = 1:4
    bestRegressorMatrix(:,i) = crossval(trainingData{:,:}, functions{i});
end

xbeta = testData{:,2:end}*bestRegressorMatrix;
y = testData{:,1};
mspe = zeros(1, 4);
for i = 1:4
    mspe(i) = (1/27)*(norm(y-xbeta(:,i)))^2;
end

trainingData = newData(1:70, [1,2,3,6,10:13]);
testData = newData(71:97, [1,2,3,6,10:13]);

newBestRegressorMatrix = zeros(7, 4);
for i = 1:4
    newBestRegressorMatrix(:,i) = crossval(trainingData{:,:}, functions{i});
end

xbeta = testData{:,2:end}*newBestRegressorMatrix;
y = testData{:,1};
newmspe = zeros(1, 4);
for i = 1:4
    newmspe(i) = (1/27)*(norm(y-xbeta(:,i)))^2;
end
