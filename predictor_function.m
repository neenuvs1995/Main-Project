function ind = predictor_function(filename,nw,numclasses,w)
% Reading image
im1 = readImage_level1(filename);
im2 = readImage_level2(filename);
im3 = readImage_level3(filename);

% Finding probabilities
p1 = activations(nw{1},im1,'softmaxL','OutputAs','rows');
p2 = activations(nw{2},im2,'softmax_L2','OutputAs','rows');
p3 = activations(nw{3},im3,'softmax_L3','OutputAs','rows');

P = [p1 ;
    p2 ;
    p3];

% Finding Certainities
I = numclasses;
C1 = max(p1) - (1/I);
C2 = max(p2) - (1/I);
C3 = max(p3) - (1/I);

P1 = [ max(p1) max(p2) max(p3)];
C = [C1 C2 C3];


% Equation4
class = sum(((w/3) .* C' .* P));
[m,ind] = max(class);

%index and max value

end
