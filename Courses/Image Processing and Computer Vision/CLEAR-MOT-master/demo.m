%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Copyright 2013 - MICC - Media Integration and Communication Center,
% University of Florence. 
% Iacopo Masi and Giuseppe Lisanti  <masi,lisanti> @dsi.unifi.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
% groundtruth and results are examples. Ricreate these two structures if
% you wanto to use it in your own multi-target tracker.


%this gives the results and the groundtruth
generateData
%threshold used to associate a tracker to a ground-truth
VOCscore = 0.5;
%display the result at the end
dispON  = true;
% run evaluation and save the result in a structure 'ClearMOT'
ClearMOT = evaluateMOT(gt,result,VOCscore,dispON);


