function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%%%%%%%%%%%%% 求cost版本一开始 %%%%%%%%%%%%%%%%%%%%%%%%%
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
	Y(i, :) = I(y(i), :);
end

a_1 = [ones(m, 1) X];
z_2 = a_1*Theta1';
a_2 = [ones(size(z_2, 1), 1) sigmoid(z_2)];
z_3 = a_2*Theta2';
a_3 = sigmoid(z_3);
hypothesis = a_3;

J = sum(sum((-Y).*log(hypothesis) - (1 - Y).*log(1-hypothesis)))/m;

J = J + lambda*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))/(2*m);

%%%%%%%%%%%%%%%%%%%%%%%%% 求cost版本一结束 %%%%%%%%%%%%%%%%%%%%%%%%%
sigma3 = a_3.-Y;
sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(size(z_2, 1), 1) z_2]);
% bias项不用归于计算中
sigma2 = sigma2(:, 2:end);

delta_1 = (sigma2'*a_1);
delta_2 = (sigma3'*a_2);

p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;



%%%%%%%%%%%%%%%%%%%%%%%%% 求cost版本二开始 %%%%%%%%%%%%%%%%%%%%%%%%%

% %part 1
% a_1 = [ones(m, 1) X];
% z_2 = a_1*Theta1';
% a_2 = [ones(m, 1) sigmoid(z_2)];
% z_3 = a_2*Theta2';
% a_3 = sigmoid(z_3);
% hypothesis = a_3;
% 
% vector_y = diag(ones(1, num_labels), 0);
% 
% for i = 1:m
% 	J = J + (-log(hypothesis(i,:))*vector_y(:,y(i)) - log(1-hypothesis(i,:))*(1-vector_y(:,y(i))));
% end
% 
% J = J/m;
% 
% 
% %2.add regularization to your cost function
% % 用这个替换循环
% J = J + lambda*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))/(2*m);

% Theta1_Total = 0;
% for i = 1:size(Theta1,1)
%     for j = 2:size(Theta1,2)
%         Theta1_Total = Theta1_Total + Theta1(i,j) ^2;
%     end
% end
% Theta1_Total
% Theta2_Total = 0;
% for i = 1:size(Theta2,1)
%     for j = 2:size(Theta2,2)
%         Theta2_Total = Theta2_Total + Theta2(i,j) ^2;
%     end
% end
% J = J + lambda * (Theta1_Total + Theta2_Total) / (2 * m);
% Theta2_Total
% 

%%%%%%%%%%%%%%%%%%%%%%%%% 求cost版本二结束 %%%%%%%%%%%%%%%%%%%%%%%%%







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
