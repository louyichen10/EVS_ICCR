%-------------------------------------------------------------------------%
% Simultaneous Estimation and Variable Selection for Interval-Censored Competing Risk Data
% LASSO, SELO, SICA, SCAD, MCP
% Choose the penalty in the Qpenalty.m function 
% and adjust the lambda candidate grid accordingly.
% Author: Lou YC
%-------------------------------------------------------------------------%
clear; clc; close all; opt = optimset('Display','off'); optfmin = optimoptions(@fminunc,'Display','off');
nrep = 100; n = 300; m = 3; obs_K = 10; emu = 1/5; nknot = 5; rhoX = 0.5;
lambset = 0.06:0.01:0.10; % To be updated with the preferred lambda candidate grid.
plambset = size(lambset, 2);
bet10 = [-0.7;-0.7;-0.7;zeros(7,1)]; bet20 = [0.0;0.0;-0.7;-0.7;-0.7;zeros(5,1)]; bet0 = [bet10; bet20];
p = size(bet10, 1); muX = zeros(1, p); sigX = eye(p);
for ii = 1:p
    for jj = 1:p
        sigX(ii,jj) = rhoX^(abs(ii-jj));
    end
end
maxIt = 1e3; maxIterk = 1e3; tol = 1e-5; bettol = 1e-5;
summary = zeros(nrep, 2*p); summaryMSE1 = zeros(nrep, 1); summaryMSE2 = zeros(nrep, 1);
summarybet_CDA = zeros(nrep, 2*p); summarybet_VAR = summarybet_CDA; 
summaryoth_CDA = zeros(nrep, 2*(m+1)); summaryMSE1_CDA = zeros(nrep, 1); summaryMSE2_CDA = zeros(nrep, 1);
for rep = 1:nrep
rng(rep)
Z = mvnrnd(muX, sigX, n);
u = random('Uniform', 0, 1, n, 1); T = fsolve(@(t) -1/3*(t.^1).*exp(Z*bet10) - 1/5.*(t.^2).*exp(Z*bet20)-log(1-u), zeros(n,1),opt);
p1 = (1/3*exp(Z*bet10)) ./ ((1/3*exp(Z*bet10))+(2/5.*(T).*exp(Z*bet20))); etype = random('Binomial',1,1-p1)+1;
L = zeros(n,1); R = L; del1 = L; del2 = L; del3 = L;
for iid = 1:n
obs = [];
while isempty(obs), obs = cumsum(random('Exp',emu,obs_K,1)); end
if T(iid) < min(obs)
    del1(iid) = 1; L(iid) = 0;  R(iid) = min(obs);
elseif T(iid) > max(obs)
    del3(iid) = 1; etype(iid) = NaN; L(iid) = max(obs); R(iid) = L(iid) + 0.01;
else
    del2(iid) = 1;
    L(iid) = max(obs(obs<=T(iid))); R(iid) = min(obs(obs>T(iid)));
end
end
l1 = 0; u1 = max(R); Laml10 = 1/3*L; Lamr10 = 1/3*R; Laml20 = 1/5*(L.^2); Lamr20 = 1/5*(R.^2); 
bl = zeros(n,(m+1)); br = bl; b0 = zeros(1,(m+1));
for i = 0:m
    bl(:,(i+1)) = bern(i,m,l1,u1,L); br(:,(i+1)) = bern(i,m,l1,u1,R); b0(:,(i+1)) =  bern(i,m,l1,u1,0);
end
phl01 = fminunc(@(x)sum((Laml10-bl*cumsum(exp(x))).^2),zeros((m+1),1), opt); phr01 = fminunc(@(x)sum((Lamr10-br*cumsum(exp(x))).^2),zeros((m+1),1), opt);
phl02 = fminunc(@(x)sum((Laml20-bl*cumsum(exp(x))).^2),zeros((m+1),1), opt); phr02 = fminunc(@(x)sum((Lamr20-br*cumsum(exp(x))).^2),zeros((m+1),1), opt);
ph01 = (phl01+phr01)/2; ph02 = (phl02+phr02)/2;
[est,~,exitflag] = fminunc(@(para) lik(para, nknot, n, m, p, l1, u1, bl, br, Z, etype, L, R), [bet10;bet20;ph01;ph02], optfmin);
summary(rep,:) = est(1:2*p); mlbet = est(1:2*p); mlothers = est((2*p+1):end);
summaryMSE1(rep,:) = (mlbet(1:p)-bet10)'*cov(Z)*(mlbet(1:p)-bet10); summaryMSE2(rep,:) = (mlbet((p+1):2*p)-bet20)'*cov(Z)*(mlbet((p+1):2*p)-bet20);
summary_lamb_bet = zeros(plambset, 2*p); summary_lamb_oth = zeros(plambset, 2*(m+1)); summary_lamb_BIC = zeros(plambset, 1);
for iilamb = 1:plambset
lamb = lambset(iilamb); bet_old = mlbet; bet = bet_old; oth_old = mlothers; oth = oth_old;
iterk = 0;
while iterk < maxIterk
    iterk = iterk+1;
    for kk = 1:(2*p)
        [betk,~,~] = fminunc(@(para) lik_bet_CDA(para, kk, bet, oth, nknot, n, m, p, l1, u1, bl, br, Z, etype, L, R, lamb), 0.1, optfmin);
        betk = (abs(betk)>bettol) * betk; bet(kk) = betk;
    end
    [oth,~,exitflagphi] = fminunc(@(para) lik_oth(para, bet, nknot, n, m, p, l1, u1, bl, br, Z, etype, L, R), oth, optfmin); deltol_bet = norm(bet - bet_old);
    if deltol_bet < tol, break; end
    bet_old = bet;
end
summary_lamb_bet(iilamb,:) = bet; summary_lamb_oth(iilamb,:) = oth;
summary_lamb_BIC(iilamb,1) = 2*lik([bet;oth], nknot, n, m, p, l1, u1, bl, br, Z, etype, L, R) + length(nonzeros(bet))*log(n);
end
[minBIC, minlambindx] = min(summary_lamb_BIC); betfinal = summary_lamb_bet(minlambindx,:); othfinal = summary_lamb_oth(minlambindx,:);
summarybet_CDA(rep,:) = betfinal; summaryoth_CDA(rep,:) = othfinal; 
summaryMSE1_CDA(rep,:) = (betfinal(1:p)'-bet10)'*cov(Z)*(betfinal(1:p)'-bet10); 
summaryMSE2_CDA(rep,:) = (betfinal((p+1):2*p)'-bet20)'*cov(Z)*(betfinal((p+1):2*p)'-bet20);
end
%%
rs = nrep; fprintf('Results:---------\n')
MMSE1 = median(summaryMSE1); MMSE1_std = std(summaryMSE1); MMSE2 = median(summaryMSE2); MMSE2_std = std(summaryMSE2);
naive_res = [MMSE1, MMSE1_std, MMSE2, MMSE2_std]; Naive_res = array2table(naive_res, VariableNames = {'MMSE1' 'STD1' 'MMSE2' 'STD2'});
disp(Naive_res); 
TP1 = mean(sum((summarybet_CDA(:,1:p)~=0).*(bet10~=0)',2)); FP1 = mean(sum((summarybet_CDA(:,1:p)~=0).*(bet10==0)',2));
TP2 = mean(sum((summarybet_CDA(:,(p+1):2*p)~=0).*(bet20~=0)',2)); FP2 = mean(sum((summarybet_CDA(:,(p+1):2*p)~=0).*(bet20==0)',2));
MMSE1 = median(summaryMSE1_CDA); MMSE1_std = std(summaryMSE1_CDA); MMSE2 = median(summaryMSE2_CDA); MMSE2_std = std(summaryMSE2_CDA);
cause1 = [TP1, FP1, MMSE1, MMSE1_std]; Cause1 = array2table(cause1, VariableNames = {'TP1' 'FP1' 'MMSE1' 'STD1'}); disp(Cause1); 
cause2 = [TP2, FP2, MMSE2, MMSE2_std]; Cause2 = array2table(cause2, VariableNames = {'TP2' 'FP2' 'MMSE2' 'STD2'});  disp(Cause2); 
SELECT_Cause1 = floor(sum(summarybet_CDA(:,1:p)~=0) * 100 / nrep);  SELECT_Cause2 = floor(sum(summarybet_CDA(:,(p+1):2*p)~=0) * 100 / nrep);
% disp(SELECT_Cause1); disp(SELECT_Cause2); 

function output = lik_oracle(para, nknot, n, m, p, l1,u1, bl, br, Z, etype, L, R)
bet1 = para(1:p); bet2 = para(p+1:2*p); 
phi1 = para(2*p+1:2*p+1+m); phi2 = para(2*p+2+m:2*p+2+2*m); ep1 = cumsum(exp(phi1)); ep2 = cumsum(exp(phi2));
Z1 = Z(:,1:3); Z2 = Z(:,3:5);
logllik = zeros(n, 1);
for ii = 1:n
   if isnan(etype(ii))
       llikii = exp(- (br(ii,:)*ep1).*exp(Z1(ii,:)*bet1) - (br(ii,:)*ep2).*exp(Z2(ii,:)*bet2));
   else
       [x, w] = Jacobi_quadrature(nknot,0,0); a = L(ii); b = R(ii);
       xt = 0.5 * (b - a) * x + 0.5 * (b + a);       
       bxt = zeros(size(xt,1),m); pbxt = bxt;
       for im = 0:m
           bxt(:,im+1) = bern(im,m,l1,u1,xt); pbxt(:,im+1) = bern_partial(im,m,l1,u1,xt);
       end
       Sall = exp(- (bxt*ep1).*exp(Z1(ii,:)*bet1) - (bxt*ep2).*exp(Z2(ii,:)*bet2));
       y1 = (pbxt*ep1) .* exp(Z1(ii,:)*bet1); y2 = (pbxt*ep2) .* exp(Z2(ii,:)*bet2);
       y = ((etype(ii)==1)*y1 + (etype(ii)==2).*y2).*Sall;
       llikii = (b - a) * dot(y,w);
   end
   logllik(ii) = log(llikii);
end
output = - sum(logllik);
end

function output = lik_oth(para, bet, nknot, n, m, p, l1,u1, bl, br, Z, etype, L, R)
bet1 = bet(1:p); bet2 = bet(p+1:2*p);
phi1 = para(1:1+m); phi2 = para(2+m:2+2*m); ep1 = cumsum(exp(phi1)); ep2 = cumsum(exp(phi2));
logllik = zeros(n, 1);
for ii = 1:n
   if isnan(etype(ii))
       llikii = exp(- (br(ii,:)*ep1).*exp(Z(ii,:)*bet1) - (br(ii,:)*ep2).*exp(Z(ii,:)*bet2));
   else
       [x, w] = Jacobi_quadrature(nknot,0,0); a = L(ii); b = R(ii);
       xt = 0.5 * (b - a) * x + 0.5 * (b + a);
       bxt = zeros(size(xt,1),m); pbxt = bxt;
       for im = 0:m
           bxt(:,im+1) = bern(im,m,l1,u1,xt); pbxt(:,im+1) = bern_partial(im,m,l1,u1,xt);
       end
       Sall = exp(- (bxt*ep1).*exp(Z(ii,:)*bet1) - (bxt*ep2).*exp(Z(ii,:)*bet2));
       y1 = (pbxt*ep1) .* exp(Z(ii,:)*bet1); y2 = (pbxt*ep2) .* exp(Z(ii,:)*bet2);
       y = ((etype(ii)==1)*y1 + (etype(ii)==2).*y2).*Sall;
       llikii = (b - a) * dot(y,w);
   end
   logllik(ii) = log(llikii);
end
output = - sum(logllik);
end

function output = lik_bet_CDA(para, k, betorg, otherspara, nknot, n, m, p, l1,u1, bl, br, Z, etype, L, R, lamborg)
betk = para; bet = betorg; bet(k) = betk; bet1 = bet(1:p); bet2 = bet(p+1:2*p);
phi1 = otherspara(1:1+m); phi2 = otherspara(2+m:2+2*m); ep1 = cumsum(exp(phi1)); ep2 = cumsum(exp(phi2));
logllik = zeros(n, 1);
for ii = 1:n
   if isnan(etype(ii))
       llikii = exp(- (br(ii,:)*ep1).*exp(Z(ii,:)*bet1) - (br(ii,:)*ep2).*exp(Z(ii,:)*bet2));
   else
       [x, w] = Jacobi_quadrature(nknot,0,0); a = L(ii); b = R(ii);
       xt = 0.5 * (b - a) * x + 0.5 * (b + a);       
       bxt = zeros(size(xt,1),m); pbxt = bxt;
       for im = 0:m
           bxt(:,im+1) = bern(im,m,l1,u1,xt); pbxt(:,im+1) = bern_partial(im,m,l1,u1,xt);
       end
       Sall = exp(- (bxt*ep1).*exp(Z(ii,:)*bet1) - (bxt*ep2).*exp(Z(ii,:)*bet2));
       y1 = (pbxt*ep1) .* exp(Z(ii,:)*bet1); y2 = (pbxt*ep2) .* exp(Z(ii,:)*bet2);
       y = ((etype(ii)==1)*y1 + (etype(ii)==2).*y2).*Sall;
       llikii = (b - a) * dot(y,w);
   end
   logllik(ii) = log(llikii);
end
pen = size(Z, 1) .* sum(Qpenalty(lamborg, bet));
output = - sum(logllik) + pen;
end

function output = lik(para, nknot, n, m, p, l1,u1, bl, br, Z, etype, L, R)
bet1 = para(1:p); bet2 = para(p+1:2*p);
phi1 = para(2*p+1:2*p+1+m); phi2 = para(2*p+2+m:2*p+2+2*m); ep1 = cumsum(exp(phi1)); ep2 = cumsum(exp(phi2));
logllik = zeros(n, 1);
for ii = 1:n
   if isnan(etype(ii))
       llikii = exp(- (br(ii,:)*ep1).*exp(Z(ii,:)*bet1) - (br(ii,:)*ep2).*exp(Z(ii,:)*bet2));
   else
       [x, w] = Jacobi_quadrature(nknot,0,0); a = L(ii); b = R(ii);
       xt = 0.5 * (b - a) * x + 0.5 * (b + a);       
       bxt = zeros(size(xt,1),m); pbxt = bxt;
       for im = 0:m
           bxt(:,im+1) = bern(im,m,l1,u1,xt); pbxt(:,im+1) = bern_partial(im,m,l1,u1,xt);
       end
       Sall = exp(- (bxt*ep1).*exp(Z(ii,:)*bet1) - (bxt*ep2).*exp(Z(ii,:)*bet2));
       y1 = (pbxt*ep1) .* exp(Z(ii,:)*bet1); y2 = (pbxt*ep2) .* exp(Z(ii,:)*bet2);
       y = ((etype(ii)==1)*y1 + (etype(ii)==2).*y2).*Sall;
       llikii = (b - a) * dot(y,w);
   end
   logllik(ii) = log(llikii);
end
output = - sum(logllik);
end

function [x, w]=Jacobi_quadrature(n,a,b)
i=1:n;
if a==0&&b==0
    del=0;
else
    del=(b^2-a^2)./((2*i+a+b).*(2*i+a+b-2));
end
i=1:n-1;
gam=2./(2*i+a+b).*sqrt(i.*(i+a+b).*(i+a).*(i+b)./((2*i+a+b).^2-1));
CM=diag(del)+diag(gam,1)+diag(gam,-1);
[V, L]=eig(CM);
[x, ind]=sort(diag(L));
V=V(:,ind)';
w= V(:,1).^2;
end

function b = bern_partial(j,p,l,u,t)
if j == 0
    b_part = p * ((1-(t-l)/(u-l)).^(p-1)) .* (-1/(u-l));
elseif j == p
    b_part = p * (((t-l)/(u-l)).^(p-1)) .* (1/(u-l));
else
    b_part = j * (1/(u-l)) * (((t-l)/(u-l)).^(j-1)).*((1-(t-l)/(u-l)).^(p-j)) + (p-j) * (-1/(u-l)) * (((t-l)/(u-l)).^j).*((1-(t-l)/(u-l)).^(p-j-1));
end
b = mycombnk(p,j) * b_part;
end

function b = bern(j,p,l,u,t)
    b = mycombnk(p,j)*(((t-l)/(u-l)).^j).*((1-(t-l)/(u-l)).^(p-j));
end

function m=mycombnk(n,k)
if nargin < 2, error('Too few input parameters'); end
s = isscalar(k) & isscalar(n);
if (~s), error('Non-scalar input'); end
ck = k > n;
if (ck), error('Invalid input'); end
z = k >= 0 & n > 0;
if (~z), error('Negative or zero input'); end
m = factorial(n)/(factorial(k)*factorial(n-k));
end 
 
