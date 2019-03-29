%% Ler base de dados com imagens dos Objetos
baseDeDadosDoObjeto = imageSet('Test','recursive');

%% Exibir montagem com os objetos representantes de cada "classe"
figure;
montage(baseDeDadosDoObjeto(1).ImageLocation);
title('Imagem do Objeto');


%%  Exibir imagem analisada e base de dados lado-a-lado
objetoParaAnalisar = 1;
galeriaDeImagens = read(baseDeDadosDoObjeto(objetoParaAnalisar),1);
figure;
for i=1:size(baseDeDadosDoObjeto,2)
listaDeImagens(i) = baseDeDadosDoObjeto(i).ImageLocation(5);
end
subplot(1,2,1);imshow(galeriaDeImagens);
subplot(1,2,2);montage(listaDeImagens);
title('Galeria de imagens');
diff = zeros(1,9);

%% Separar grupo de treino e grupo de teste
[treinamento,teste] = partition(baseDeDadosDoObjeto,[0.8 0.2]);


%% Extrair e exibir o Histograma de Gradiente Orientado de um objeto
objeto = 5;
[hdoCarac, visualisacao]= extractHOGFeatures(read(treinamento(objeto),1));
figure;
subplot(2,1,1);imshow(read(treinamento(objeto),1));title('Objeto de Entrada');
subplot(2,1,2);plot(visualisacao);title('Histograma de Gradiente Orientado');

%% Extrair HGO do grupo de treinamento
grupoDeTreino = zeros(size(treinamento,2)*treinamento(1).Count,4356);
contadorDeobj = 1;
for i=1:size(treinamento,2)
    for j = 1:treinamento(i).Count
        grupoDeTreino(contadorDeobj,:) = extractHOGFeatures(read(treinamento(i),j));
        rotuloDeTreino{contadorDeobj} = treinamento(i).Description;    
        contadorDeobj = contadorDeobj + 1;
    end
    indiceDoObj{i} = treinamento(i).Description;
end

%% Criar 40 classes com o fitcecoc 
classificadorDeObj = fitcecoc(grupoDeTreino,rotuloDeTreino);

%% Testar primeiros 10 objetos do Grupo de Teste
figure;
figureNum = 1;
for objeto=1:40
    for j = 1:teste(objeto).Count
        imagemAnalisada = read(teste(objeto),j);
        caracAnalisada = extractHOGFeatures(imagemAnalisada);
        personLabel = predict(classificadorDeObj,caracAnalisada);
        % Mapear conjunto de treinamento para encontrar identidade
        indiceBoleano = strcmp(personLabel, indiceDoObj);
        indiceInteiro = find(indiceBoleano);
        subplot(2,2,figureNum);imshow(imresize(imagemAnalisada,3));title('Objeto de Consulta');
        subplot(2,2,figureNum+1);imshow(imresize(read(treinamento(indiceInteiro),1),3));title('Classe Atribuída');
        figureNum = figureNum+2;
        
    end
    figure;
    figureNum = 1;

end