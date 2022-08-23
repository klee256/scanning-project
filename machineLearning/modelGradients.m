function [loss,infGrad] = modelGradients(encoderNet,x,mat) 
    z=forward(encoderNet,x,Outputs=["fc_1"]);
    z=sigmoid(z);
    loss=mse(z,mat);
    infGrad=dlgradient(loss,encoderNet.Learnables);
end
