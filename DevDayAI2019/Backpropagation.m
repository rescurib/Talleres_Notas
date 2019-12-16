%Carga de datos de entrenamineto
%load('dataset_BP_blobs.mat')
load('dataset_BP_moons.mat')

% Red neuronal de 3 capas: 2|4|2
%--- Pesos ---
xmin=-0.25;
xmax=0.25;
W1=xmin+rand(2,4)*(xmax-xmin) %Pesos capa 1
W2=xmin+rand(4,2)*(xmax-xmin) %Pesos capa 1
b1=xmin+rand(4,1)*(xmax-xmin) %Pesos capa 1
b2=xmin+rand(2,1)*(xmax-xmin) %Pesos capa 1
%-----------
%--- Entrenamiento ---
t = dataset(:,3:4);  %target
p = dataset(:,1:2); %patrones
a = 0.5;
for Ep = 1:1000
    for k = 1:size(t,1)
        net1 = W1'*p(k,:)'+b1;
        h = logsig(net1);
        net2 = W2'*h+b2;
        y = logsig(net2);
        e = t(k,:)' - y;
        
        df1 = logsig(net1).*(1-logsig(net1));
        df2 = logsig(net2).*(1-logsig(net2));
        
        S2 = e.*df2;
        S1 = (W2*S2).*df1;
        
        %--- Actualización de W2 ---
        for i = 1:size(W2,1)   %i = 1:No de neuronas en L2
            for j=1:size(W2,2) %j = 1:No. de neuronas en L1
                W2(i,j) = W2(i,j) + a*S2(j).*h(i);
            end
        end
        
        b2 = b2 + a*S2; % Bias capa L2
        
        %--- Actualizar W1 ---
        for i = 1:size(W1,1)     %i = 1:No de neuronas en L1
            for j = 1:size(W1,2) %j = 1:No. de entradas en L0
                W1(i,j) = W1(i,j) + a*S1(j).*p(k,i);
            end
        end
        
        b1 = b1 + a*S1; % Bias capa L1
        
    end
end


%% Graficas
%Mapa de predicción
x0 = linspace(min(dataset(:,1)),max(dataset(:,1)));
x1 = linspace(min(dataset(:,2)),max(dataset(:,2)));
[X0,X1] = meshgrid(x0,x1);

for i = 1:size(X0,1)
    for j = 1:size(X0,2)
        %Modelo
        net1 = W1'*[X0(i,j),X1(i,j)]'+b1;
        h = logsig(net1);
        net2 = W2'*h+b2;
        y = logsig(net2);
        Y(i,j) = round(y(1));
    end
end

contourf(X0,X1,Y,10)
map =[0 0 1
      1 0 0];
colormap(map)

hold on
for i = 1:size(dataset,1)
    if(dataset(i,3)==1)
        scatter(dataset(i,1),dataset(i,2),'MarkerEdgeColor','k','MarkerFaceColor','r')
    else
        scatter(dataset(i,1),dataset(i,2),'MarkerEdgeColor','k','MarkerFaceColor','b')
    end
end

xlabel('x')
ylabel('y')
grid on 
