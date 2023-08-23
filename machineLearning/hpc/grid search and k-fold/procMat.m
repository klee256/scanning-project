% procMat.m

function out_mat = procMat(in_mat,n)

    dim=sqrt(size(in_mat,2));
    startingPos=n+1;
    endingPos=dim-n;
    out_mat = windowPlot(in_mat,startingPos,endingPos);

    