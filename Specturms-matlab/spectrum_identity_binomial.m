% Machenro Pastu distribution 
% Wishart Matrix initialized with identity matrix 
% Author: Wenping Cui, March, 2019.
clear all;
close all;
N=100;
Sample_size=5000;
Sig=[0.001, 0.01, 0.02, 0.1];
C=[0.96,0.89,0.96,0.97]; % c is obtained from numerical simulations
for i=1:4
    sigc=Sig(i)
    c=C(i);
    L=zeros(Sample_size,N*c);
    for j=1:Sample_size
        x=rand(N,N) < sigc;
        x=x+1*eye(N);
        x=x(1:c*N,:);
        s=std(x(:))*sqrt(N);
        a=(s^2)*(1-sqrt(c))^2;
        b=(s^2)*(1+sqrt(c))^2;
        M=x*x'; % M^T=M.
        L(j,:)=eig(M);
    end
if i>2
    b=b+2;
    Nbins=1000;
    [Y,X]=hist(L,linspace(0,b,Nbins));
else 
    Nbins=100;
    [Y,X]=hist(L,linspace(a,b+5,Nbins));
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
if i== 3
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
    axis([0 10 0 0.02]);
    legend(h2,'Marchenko-Pastur distribution','FontSize',20)
    title(strcat('p_c=',num2str(sigc)),'FontSize',40);
elseif i== 4
    h2=plot(lambda,F,'r','LineWidth',2);
    hold off;
    axis([0 10 0 0.02]);
    legend(h2,'Marchenko-Pastur distribution','FontSize',20)
    title(strcat('p_c=',num2str(sigc)),'FontSize',40);
elseif i==1
    axis([0 10 0 1.0]);
    title(strcat('p_c=',num2str(sigc)),'FontSize',40);
elseif  i==2
    axis([0 10 0 0.2]);
    title(strcat('p_c=',num2str(sigc)),'FontSize',40);
elseif  i==3
    axis([0 10 0 0.1]);
    title(strcat('p_c=',num2str(sigc)),'FontSize',40);
end
xlabel('\lambda','FontWeight','bold','FontSize',30)
ylabel('PDF(\lambda)','FontWeight','bold','FontSize',30);
fig = gcf;
ax = gca;
ax.FontSize = 30;
fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
saveas(fig,strcat('Binary_p_',num2str(sigc),'_c',num2str(c),'.pdf'),'pdf');
end