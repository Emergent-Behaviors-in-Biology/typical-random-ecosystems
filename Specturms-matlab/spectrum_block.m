% Machenro Pastu distribution 
% Wishart Matrix initialized with block matrix
% gaussian noise
% Author: Wenping Cui, March, 2019.
clear all;
close all;
N=100;
Sample_size=1000;
sigc=1.5;
B=kron(eye(10),repmat(1,10,10));
Sig=[0., 1, 5, 10.0, 20];
C=[1., 0.92,0.97,0.96,0.9]; % c is obtained from numerical simulations
for i=1:5
c=C(i);
L=zeros(Sample_size,N*c);
sigc=Sig(i);
for j=1:Sample_size
x=sigc/sqrt(N)*randn(N,N)+B;
x=x(1:c*N,:);
s=std(x(:))*sqrt(N);
a=(s^2)*(1-sqrt(c))^2;
b=(s^2)*(1+sqrt(c))^2;
M=x*x'; % M^T=M.
L(j,:)=eig(M);
end
Nbins=200;
if i>3
    b=b+40;
    [Y,X]=hist(L,linspace(a,b,Nbins));
elseif i==1
    [Y,X]=hist(L,linspace(0,20,Nbins));
    a=0;
    b=80;
else 
    [Y,X]=hist(L,Nbins);
end
Y=Y/sum(Y);
figure
map = brewermap(3,'Set1');
h=bar(X,Y,'facecolor',map(1,:),'facealpha',.4,'edgecolor','none');
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
hold on;
% Theoretical Probability density function
r=max(L);
lambda=linspace(a,b,Nbins);
ft=@(lambda,a,b,c) (1./(2*pi*lambda*c*s^(2))).*sqrt((b-lambda).*(lambda-a));
F=ft(lambda,a,b,c);
F(isnan(F))=0;
F=F/sum(F);
if i==4 | i==5
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
    %axis([0 8 0 0.07]);
    legend(h2,'Marchenko-Pastur distribution','FontSize',20)
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',40);
elseif i==1
    axis([0 40 0 0.2]);
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',40);
elseif i==2
    %axis([0 2.5 0 0.018]);
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',40);
elseif i==3
    %axis([0 5 0 0.025]);
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',40);
end
xlabel('\lambda','FontWeight','bold','FontSize',30)
ylabel('PDF(\lambda)','FontWeight','bold','FontSize',30);
fig = gcf;
ax = gca;
ax.FontSize = 30;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
saveas(fig,strcat('Block_sigma_c_',num2str(sigc),'.pdf'),'pdf');
end

    