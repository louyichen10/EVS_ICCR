function output = Qpenalty(lamb, betk)
%% SCAD
a  = 3.7;
output = (abs(betk) <= lamb) .* (lamb.*abs(betk)) + ...
    (abs(betk) >= a*lamb) .* ((a+1)/2*lamb^2) + ...
    (abs(betk) > lamb) .* (abs(betk) <= a*lamb) .* ...
    (- (betk.^2 - 2*a*lamb.*abs(betk) + lamb^2) / (2*(a-1)));
%% MCP
% gam = 1.4;
% output = (abs(betk) <= gam*lamb) .* (lamb.*abs(betk) - betk.^2/(2*gam)) + ...
%     (abs(betk) > gam*lamb) .* (1/2*gam*lamb^2);
%% SICA
% gam = 0.01;
% output = lamb .* ( ((gam+1).*abs(betk)) ./ (abs(betk) + gam) );
%% SELO
% gam = 0.01;
% output = lamb/log(2) .*log ( 1 + abs(betk) ./ (abs(betk) + gam) );
%% LASSO
% output = lamb .* abs(betk);
