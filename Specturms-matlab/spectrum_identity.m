% Machenro Pastu distribution 
% Wishart Matrix initialized with identity matrix 
% gaussian noise
% Author: Wenping Cui, March, 2019.
clear all;
close all;
N=100;
Sample_size=1000;
sigc=1.5;
Sig=[0.01,0.1, 0.3, 0.7, 1.0, 10.]; % c is obtained from numerical simulations
C=[1.,1.,1., 0.94, 0.92, 0.84];
for i=1:6
    if i==6
        Sample_size=100000;
    end
    sigc=Sig(i);
    c=C(i);
    L=zeros(Sample_size,N*c);
    B=eye(N);
    for j=1:Sample_size
        x=sigc/sqrt(N)*randn(N*c,N)+B(1:N*c,:);
        s=std(x(:))*sqrt(N);
        a=(s^2)*(1-sqrt(c))^2;
        b=(s^2)*(1+sqrt(c))^2;
        M=x*x'; % M^T=M.
        L(j,:)=eig(M);
    end
Nbins=100;
if i==6
    Nbins=10000;
    [Y,X]=hist(L,linspace(0,b,Nbins));
elseif i==4
    Nbins=1000;
    [Y,X]=hist(L,linspace(0,5,Nbins));
else 
    [Y,X]=hist(L,Nbins);
end
min(L(:))
Y=Y/sum(Y);
figure
map = brewermap(3,'Set1');
h=bar(X,Y,'facecolor',map(1,:),'facealpha',.4,'edgecolor','none');
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
hold on;
% Theoretical Probability density function
lambda=linspace(a,b,Nbins);
ft=@(lambda,a,b,c) (1./(2*pi*lambda*c*s^(2))).*sqrt((b-lambda).*(lambda-a));
F=ft(lambda,a,b,c);
F(isnan(F))=0;
F=F/sum(F);
if sigc==1.0
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
    axis([0 10 0 0.07]);
    %legend(h2,'Marchenko-Pastur distribution y=1.0','FontSize',20)
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif i==6
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
    axis([0 10 0 0.001]);
    %legend(h2,'Marchenko-Pastur distribution','FontSize',20)
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif sigc==0.01
    axis([0 10 0 0.015]);
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif sigc==0.1
    axis([0 10 0 0.015]);
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif sigc==0.3
    axis([0 10 0 0.018]);
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif sigc==0.7
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
    axis([0 5 0 0.016]);
    legend(h2,'Marchenko-Pastur distribution','FontSize',20)
    title(strcat("Regime B ",'\sigma_c=',num2str(sigc)),'FontSize',20);
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
end
xlabel('\lambda','FontWeight','bold','FontSize',30)
ylabel('PDF(\lambda)','FontWeight','bold','FontSize',30);
fig = gcf;
ax = gca;
ax.FontSize = 30;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
saveas(fig,strcat('Identity_sigma_c_',num2str(sigc),'.pdf'),'pdf');
end