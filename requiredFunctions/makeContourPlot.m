function makeContourPlot(dataArray,userYlabel,userTitle,step,flip)

    % flip refers to flipping the color map

    dim=sqrt(length(dataArray));
    dataMat=reshape(dataArray,dim,dim);
    [XX,YY]=meshgrid(linspace(0,dim*step,dim)); 
    %FigH = figure('Position', get(0, 'Screensize'));
    figure();
    contourf(XX,YY,dataMat);
    xlim([0 dim*step]); ylim([0 dim*step]);
    axis square
    %tickMark=linspace(0,2,7);
    tickMark=[0 0.4 0.8 1.2 1.6 2];
    xticks(tickMark);
    yticks(tickMark);
    hc=colorbar;
    hc.Label.String=userYlabel;
    title(userTitle)
    xlabel('Position (mm)'); ylabel('Position (mm)');
    if (flip == 'y')
        colormap(flipud(turbo))
    else
        colormap(turbo)
    end
    set(gca,'FontSize',18)

    %exportgraphics(gca,'cap3.png')

    % To plot rectangular:
    % first col controls Y in XX(1:70,1:35) YY(1:70,1:35)
    
    % wholeN=reshape(n,70,70);
    % figure();
    % [XX,YY]=meshgrid(linspace(0,70*0.025,70)); 
    % contourf(XX(1:35,1:70),YY(1:35,1:70),wholeN(1:35,1:70));


