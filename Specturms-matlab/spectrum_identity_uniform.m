% Machenro Pastu distribution 
% Wishart Matrix initialized with identity matrix 
% gaussian noise
% Author: Wenping Cui, March, 2019.
clear all;
close all;
N=10;
Sample_size=1;
sigc=1.5;
Sig=[0.01, 0.05, 0.1, 0.4 0.5, 0.6,0.7,0.8, 1.0, 5.]; % c is obtained from numerical simulations
Sig=[0, 0.4 0.5, 0.6,0.7,0.8, 1.0, 5.];
C1=[1.,1., 1, 1, 1, 1,1,1]*0.4;
C2=[1.,1., 1, 1, 1, 1,1,1];
for i=1:1
    sigc=Sig(i);
    c1=C1(i);
    c2=C2(i);
    L=zeros(Sample_size,N*c1);
    B=eye(N);
    for j=1:Sample_size
        %gaussian
        x=sigc/sqrt(N)*randn(N,N)+B+1.;
        % uniform
       % x=sigc*rand(N,N)+B;
        c=c1/c2;
        out1=randperm(N);
        ri1=out1(1:c1*N);
        out2=randperm(N);
        ri2=out2(1:c2*N);
        x=x(sort(ri1),sort(ri2));
        s=sqrt(sigc^2/12*100);
        s=std(x(:))*sqrt(N);
        a=(s^2)*(1-sqrt(c))^2;
        b=(s^2)*(1+sqrt(c))^2;
        M=x*x'; % M^T=M.
        L(j,:)=eig(M);
    end
L(L>b^2+10)=0;
Nbins=100;
[Y,X]=hist(L,Nbins);
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
    %legend(h2,'Marchenko-Pastur distribution y=1.0','FontSize',20)
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif i==6
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
    %legend(h2,'Marchenko-Pastur distribution','FontSize',20)
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif sigc==0.01
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif sigc==0.1
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif sigc==0.5
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
    title(strcat('\sigma_c=',num2str(sigc)),'FontSize',60);
elseif sigc==0.7
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
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