# All notebooks for master degree by chapter
This folder contains all code for the master thesis, chapter for chapter with elaborations on concepts and illustrations. 
For a in-depth description  of the concepts, you would have to read the actual thesis. 

Most code is created in the /scripts folder. 

The notebooks follow the following order:

01. Data analysis 
...A in-depth analysis of seismic from a visual (computer vision) perspective 
02. (1) Data sequence analysis (image)
... Look at how 3d volumetric macroscopic data patterns evolve across the survey in all 3 directions from a image perspective. Here we spread out each image in the 3d sequence, for then to lay them out based on how differnt they are. Much like laying a stack of cards on a table, for then to stack those of similar color/number toghether in seperate stacks. 
02. (2) Data sequence analysis (patches)
... Look at how 3d volumetric microscopic data patterns evolve across the survey in all 3 directions from a more granular patch perspective. Here we lay them out the same way as with images, and reveal more evident stacking patterns. 
02. (2.1) (Use case) Patch retrieval
... As a user  want to picka  part of a slice, and find all similar patches in the cube. I want a overview over where these patches are  image wise, and patchwise withing the image. In addition i want to provide a treshold for how far away the similar image can be, to prevent only getting next-image/wiring-image neighboring patches. 
03. Quantifying similarity using clustering on PCA, T-SNE and UMAP
... We now have a overview over micro and macro patterns. We now look at how we can quantify these similarities by clustering them in low-dimensionality space by first projecting them to 2d coordinates, and then clustering them using clustering algorithms.  Here we need to be carefull, as these techinques are only valid for visualization. If one are aware of pitfalls, tune hyperparameter like perplexity correctly, it will be highly efficient to locate and veryfy patterns, but it cannot be used for density/distance computation. In conclusions, use t-SNE for visualization (and try different parameters to get something visually pleasing!), but rather do not run clustering afterwards, in particular do not use distance- or density based algorithms, as this information was intentionally (!) lost. Neighborhood-graph based approaches may be fine, but then you don't need to first run t-SNE beforehand, just use the neighbors immediately (because t-SNE tries to keep this nn-graph largely intact). Umap is a newer alternative and is better suited as it keeps meaningful sitanses ovto the latent manifold
https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne
https://distill.pub/2016/misread-tsne/
03. interpolation in latent space
... to get a feel about hte projected latence space, we can interpolate aloong points to get a feel for the distribution
03. Quantifying similarity using similarity learning
... To match an encoding (that be pixel or laten space) to another one, we need to compare both representation and provide a resutling metric. this is solved in similarity learning.........
03. Quantifying similarity using distanse/density based algorithm 
... For a less trivial task, we want to cluster arbitrary similar representations toghether, maintaining the distance between (that T_SNE does not) so it can be used for similarity retrieval. Here we look to automaticly detect clusters of similar characteristics based on their density/distance metrics. If we can infer the distribution in feature space, we can supsample more efficienly for multiple purposes (e.g trianing data)
03. Quantifying similarity suing active learning
... file:///Users/anderskampenes/Downloads/Distance_in_Latent_Space_as_Novelty_Measure.pdf indicates that the labels of a dataset should be sparsely distributed over the entire feature space

03. Geometrical Aspects of Manifold Learning
... One of the most common assumptions in machine learning is that the data lie
near a non-linear low dimensional manifold in the ambient data space. In this
setting, machine learning methods should be developed in such a way to respect
the underlying geometric structure implied by the manifold. When the data lie
on such a curved space, then the shortest path between two points is actually
a curve on the manifold and its length is the most natural distance measure.
Hence, a suitable way to measure such distances is to model the curved space
as a Riemannian manifold. The reason is that by learning a Riemannian metric
in the ambient space where the data lives, we transform this Euclidean space
into a Riemannian manifold, which enables us to compute the shortest path. In
this thesis, we study how statistical models can be generalized on these curved
spaces, also we develop methods to learn the Riemannian manifold directly from
the data, as well as an efficient method to compute shortest paths.
04. Feature representation
... Instead of only operating in pixel space, we can use more advanced teqniques to find better representations for each slice/patch. These will capture more important/promenant features. Among such methods are CPC, encoders and seismic attributes.  We first try a disciminative ( is it discriminative with autoregressor??) model using cpc, and compare it to the generative alternative VAE (and GAN??) to encode our image ono a laten t manifold. Here we make copmparison on the autoregressive generativemodela nd the generative VAE/GAN, to see whe better represent the distances. 
05. Defining target structures using clustering technicques.
... Until now we have only looked at similarity based on arbitrary characteristics. Most often, one is interested in spesific target structures, and therefore we want to use the methodology from the previous chapters enable this. 
06. Information/image retieaval 
... To verify the distance metrics we create informatioal retrieval algorithm to rank similarity/didsiibmilaroyt based on the ,etric learned. This way we can veridy qualitativly  and quantativly (if we have labels) hopw well the distance metric is. 

should we use the wasserstein loss for distancing?????