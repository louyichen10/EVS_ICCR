%-------------------------------------------------------------------------%
% Application - Framingham
% Author: Lou Y.C.
%-------------------------------------------------------------------------%
clear; clc; close all; rng(1)
lambset = 0.0005:0.0005:0.004; % To be updated with the preferred lambda candidate grid.
plambset = size(lambset, 2); Framingham = importfile("Framingham.csv", [2, Inf]); 
T = Framingham.time_outcome; L = floor(T)/10; R = ceil(T)/10; etype = Framingham.outcome;
del1 = (L==0); del3 = (etype==0); del2 = 1-del1-del3; etype(etype==0) = NaN;
Totchol = Framingham.totchol; Age = Framingham.age; BMI = Framingham.bmi; Female = 2*Framingham.female-1;
BPVAR = Framingham.BPVar; Heartrte = Framingham.heartrte; Glucose = Framingham.glucose; Cigpday = Framingham.cigpday;
Totchol = (Totchol - mean(Totchol)) ./ std(Totchol); Age = (Age - mean(Age)) ./ std(Age); BMI = (BMI - mean(BMI)) ./ std(BMI);
BPVAR = (BPVAR - mean(BPVAR)) ./ std(BPVAR); Heartrte = (Heartrte - mean(Heartrte)) ./ std(Heartrte); 
Glucose = (Glucose - mean(Glucose)) ./ std(Glucose); Cigpday = (Cigpday - mean(Cigpday)) ./ std(Cigpday);
Z = [Female, Totchol, Age, BMI, BPVAR, Heartrte, Glucose, Cigpday]; p = size(Z, 2); n = size(Z, 1);
m = 6; nknot = 8; maxIt = 1e3; maxIterk = 1e3; tol = 1e-5;  bettol = 1e-5; 
summary = zeros(1, 2*p); summarybet_CDA = zeros(1, 2*p); summaryoth_CDA = zeros(1, 2*(m+1));
opt = optimset('Display','off');
optfmin = optimoptions(@fminunc,'Display','off','MaxIterations',1e6,'MaxFunctionEvaluations', 1e6,'StepTolerance',1e-6);
l1 = 0; u1 = max(R); Laml10 = 1/3*L; Lamr10 = 1/3*R; Laml20 = 1/5*(L.^2); Lamr20 = 1/5*(R.^2);
bl = zeros(n,(m+1)); br = bl; b0 = zeros(1,(m+1));
for i = 0:m
    bl(:,(i+1)) = bern(i,m,l1,u1,L); br(:,(i+1)) = bern(i,m,l1,u1,R); b0(:,(i+1)) =  bern(i,m,l1,u1,0);
end
phl01 = fminunc(@(x)sum((Laml10-bl*cumsum(exp(x))).^2),zeros((m+1),1), opt); phr01 = fminunc(@(x)sum((Lamr10-br*cumsum(exp(x))).^2),zeros((m+1),1), opt);
phl02 = fminunc(@(x)sum((Laml20-bl*cumsum(exp(x))).^2),zeros((m+1),1), opt); phr02 = fminunc(@(x)sum((Lamr20-br*cumsum(exp(x))).^2),zeros((m+1),1), opt);
ph01 = (phl01+phr01)/2; ph02 = (phl02+phr02)/2;
[est,~,exitflag,~,~,hesl] = fminunc(@(para) lik(para, nknot, n, m, p, l1, u1, bl, br, Z, etype, L, R, 0, 1), [zeros(p,1);zeros(p,1);ph01;ph02], optfmin);
bet1final = est(1:p); bet2final = est((p+1):(2*p));
try
    diagg = diag(pinv(hesl)); dihh = sqrt(abs(diagg(1:(2*p)))); summvar = dihh;
catch
    summvar = nan(1,(2*p));
end
mlbet = est(1:2*p); mlothers = est((2*p+1):end);
summary_lamb_bet = zeros(plambset, 2*p); summary_lamb_oth = zeros(plambset, 2*(m+1)); summary_lamb_BIC = zeros(plambset, 1);
for iilamb = 1:plambset
    lamb = lambset(iilamb); bet_old = mlbet; bet = bet_old; oth_old = mlothers; oth = oth_old;
    iterk = 0;
    while iterk < maxIterk
        iterk = iterk+1;
        for kk = 1:(2*p)
            [betk,~,~] = fminunc(@(para) lik_bet_CDA_ALASSO(para, kk, bet, oth, nknot, n, m, p, l1, u1, bl, br, Z, etype, L, R, lamb, mlbet(kk), 0, 1), 0.0, optfmin);
            betk = (abs(betk)>bettol)*betk; bet(kk) = betk;
        end
        [oth,~,exitflagphi] = fminunc(@(para) lik_oth(para, bet, nknot, n, m, p, l1, u1, bl, br, Z, etype, L, R, 0, 1), oth, optfmin); deltol_bet = norm(bet - bet_old);
        if deltol_bet < tol, break; end
        bet_old = bet;
    end
    summary_lamb_bet(iilamb,:) = bet; summary_lamb_oth(iilamb,:) = oth;
    summary_lamb_BIC(iilamb,1) = 2*lik([bet;oth], nknot, n, m, p, l1, u1, bl, br, Z, etype, L, R, 0, 1) + length(nonzeros(bet))*log(n);
end
[minBIC, minlambindx] = min(summary_lamb_BIC); betfinal = summary_lamb_bet(minlambindx,:); othfinal = summary_lamb_oth(minlambindx,:);
try
    hessfinal = hessian(@(para) lik(para, nknot, n, m, p, l1, u1, bl, br, Z, etype, L, R, 0, 1), [betfinal';othfinal']);
    diagg = diag(pinv(hessfinal)); dihh = sqrt(abs(diagg(1:(2*p)))); summvarVS = dihh;
catch
    summvarVS = nan(1,(2*p));
end
bet1finalVS = betfinal(1:p); bet2finalVS = betfinal((p+1):(2*p));
fprintf('Results:\n')
betfinal = [bet1finalVS, bet2finalVS]'; betdih = summvarVS; pval1 = normcdf(-betfinal./betdih)*2;
res1 = [betfinal,betdih, min(pval1, 2*(1-pval1./2))];
allresSHOW_table = array2table(["Female", "Totchol", "Age", "BMI", "BPVAR", "Heartrte", "Glucose", "Cigpday"]', "VariableNames","Factor"); res1SHOW = res1(1:p,:);
RES_table = array2table(res1SHOW, VariableNames = {'EST' 'SD' 'PVAL'}); RES1 = [allresSHOW_table, RES_table];
res2SHOW = res1((p+1):end,:); RES_table = array2table(res2SHOW, VariableNames = {'EST' 'SD' 'PVAL'}); RES2 = [allresSHOW_table, RES_table];
disp(RES1); disp(RES2);

function output = lik(para, nknot, n, m, p, l1,u1, bl, br, Z, etype, L, R, uu1, uu2)
bet1 = para(1:p); bet2 = para(p+1:2*p);
phi1 = para(2*p+1:2*p+1+m); phi2 = para(2*p+2+m:2*p+2+2*m); ep1 = cumsum(exp(phi1)); ep2 = cumsum(exp(phi2));
logllik = zeros(n, 1);
for ii = 1:n
   if isnan(etype(ii))
       llikii = exp(- (br(ii,:)*ep1).*exp(Z(ii,:)*bet1) - (br(ii,:)*ep2).*exp(Z(ii,:)*bet2));
   else
       [x, w] = Jacobi_quadrature(nknot,uu1,uu2); a = L(ii); b = R(ii);
       xt = 0.5 * (b - a) * x + 0.5 * (b + a);       
       bxt = zeros(size(xt,1),m); pbxt = bxt;
       for im = 0:m
           bxt(:,im+1) = bern(im,m,l1,u1,xt); pbxt(:,im+1) = bern_partial(im,m,l1,u1,xt);
       end
       Sall = exp(- (bxt*ep1).*exp(Z(ii,:)*bet1) - (bxt*ep2).*exp(Z(ii,:)*bet2));
       y1 = (pbxt*ep1) .* exp(Z(ii,:)*bet1); y2 = (pbxt*ep2) .* exp(Z(ii,:)*bet2);
       y = ((etype(ii)==1)*y1 + (etype(ii)==2).*y2).*Sall;
       llikii = dot(y,w);
   end
   logllik(ii) = log(llikii);
end
output = - sum(logllik);
end

function output = lik_bet_CDA_ALASSO(para, k, betorg, otherspara, nknot, n, m, p, l1,u1, bl, br, Z, etype, L, R, lamborg, betk_tilde, uu1, uu2)
betk = para; bet = betorg; bet(k) = betk; bet1 = bet(1:p); bet2 = bet(p+1:2*p);
phi1 = otherspara(1:1+m); phi2 = otherspara(2+m:2+2*m); ep1 = cumsum(exp(phi1)); ep2 = cumsum(exp(phi2));
logllik = zeros(n, 1);
for ii = 1:n
   if isnan(etype(ii))
       llikii = exp(- (br(ii,:)*ep1).*exp(Z(ii,:)*bet1) - (br(ii,:)*ep2).*exp(Z(ii,:)*bet2));
   else
       [x, w] = Jacobi_quadrature(nknot,uu1,uu2); a = L(ii); b = R(ii);
       xt = 0.5 * (b - a) * x + 0.5 * (b + a);
       bxt = zeros(size(xt,1),m); pbxt = bxt;
       for im = 0:m
           bxt(:,im+1) = bern(im,m,l1,u1,xt); pbxt(:,im+1) = bern_partial(im,m,l1,u1,xt);
       end
       Sall = exp(- (bxt*ep1).*exp(Z(ii,:)*bet1) - (bxt*ep2).*exp(Z(ii,:)*bet2));
       y1 = (pbxt*ep1) .* exp(Z(ii,:)*bet1); y2 = (pbxt*ep2) .* exp(Z(ii,:)*bet2);
       y = ((etype(ii)==1)*y1 + (etype(ii)==2).*y2).*Sall;
       llikii = dot(y,w);
   end
   logllik(ii) = log(llikii);
end
pen = size(Z, 1) .* sum(Qpenalty_ALASSO(lamborg, bet, betk_tilde));
output = - sum(logllik) + pen;
end

function output = lik_oth(para, bet, nknot, n, m, p, l1,u1, bl, br, Z, etype, L, R, uu1, uu2)
bet1 = bet(1:p); bet2 = bet(p+1:2*p);
phi1 = para(1:1+m); phi2 = para(2+m:2+2*m); ep1 = cumsum(exp(phi1)); ep2 = cumsum(exp(phi2));
logllik = zeros(n, 1);
for ii = 1:n
   if isnan(etype(ii))
       llikii = exp(- (br(ii,:)*ep1).*exp(Z(ii,:)*bet1) - (br(ii,:)*ep2).*exp(Z(ii,:)*bet2));
   else
       [x, w] = Jacobi_quadrature(nknot,uu1,uu2); a = L(ii); b = R(ii);
       xt = 0.5 * (b - a) * x + 0.5 * (b + a);
       bxt = zeros(size(xt,1),m); pbxt = bxt;
       for im = 0:m
           bxt(:,im+1) = bern(im,m,l1,u1,xt); pbxt(:,im+1) = bern_partial(im,m,l1,u1,xt);
       end
       Sall = exp(- (bxt*ep1).*exp(Z(ii,:)*bet1) - (bxt*ep2).*exp(Z(ii,:)*bet2));
       y1 = (pbxt*ep1) .* exp(Z(ii,:)*bet1); y2 = (pbxt*ep2) .* exp(Z(ii,:)*bet2);
       y = ((etype(ii)==1)*y1 + (etype(ii)==2).*y2).*Sall;
       llikii = dot(y,w);
   end
   logllik(ii) = log(llikii);
end
output = - sum(logllik);
end

function b = bern(j,p,l,u,t)
    b = mycombnk(p,j)*(((t-l)/(u-l)).^j).*((1-(t-l)/(u-l)).^(p-j));
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

function output = Qpenalty_ALASSO(lamb, betk, betk_tilde)
output = lamb .* abs(betk) ./ abs(betk_tilde);
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
 
