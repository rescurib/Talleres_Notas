%Carga de datos de entrenamiento
load('desgrad_dataset.mat')

%Neurona artificial
% x[0]--w[1]--
%              \
%                + --f(net)-- y
%              /
% x[1]--w[1]--

%Inicializacion de pesos (aleatorios)
xmin=-1
xmax=1
w=xmin+rand(1,2)*(xmax-xmin) %vector de pesos dimencion [1,2]
b = rand() % bias


%% Entrenamiento
t = dataset(:,3);  %target
p = dataset(:,1:2); %patrones

N = 100; %Numero de epocas de entrenamiento
a = 0.1; %Factor de aprendisaje
for Ep = 1:N
    for k = 1:numel(t)
        net = dot(w,p(k,:))+b;
        y = logsig(net);
        df = logsig(net)*(1-logsig(net));
        e = t(k)- y;
        w = w + a*e*df*p(k,:);
        b = b + a*e*df;
    end
end

%% Graficas
%Mapa de prediccion
x0 = linspace(-1,7);
x1 = linspace(-1,7);
[X0,X1] = meshgrid(x0,x1);

for i = 1:size(X0,1)
    for j = 1:size(X0,2)
        Y(i,j) = logsig(dot(w,[X0(i,j),X1(i,j)])+b);
    end
end

contourf(X0,X1,Y,10)
map =[0 0 1
      1 0 0];
colormap(map)
%%
hold on
for i = 1:size(dataset,1)
    if(dataset(i,3)==1)
        scatter(dataset(i,1),dataset(i,2),'MarkerEdgeColor','k','MarkerFaceColor','r')      
    else
        scatter(dataset(i,1),dataset(i,2),'MarkerEdgeColor','k','MarkerFaceColor','b')
    end
end
hold off
xlabel('x')
ylabel('y')
grid on 
