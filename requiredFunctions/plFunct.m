function [peakWavelength,amplitudes] = plFunct(plDataRaw)

    
    peakWavelength=[];
    amplitudes=[];
    wavelengths=linspace(1050,1700,512)';
    for i=1:length(plDataRaw)
        [maxA,maxI]=max(plDataRaw(:,i));
        amplitudes=cat(2,amplitudes,maxA);
        peakWavelength=cat(2,peakWavelength,wavelengths(maxI));
    end