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
03. Quantifying similarity using clustering on PCA and T-SNE
... We now have a overview over micro and macro patterns. We now look at how we can quantify these similarities by clustering them in low-dimensionality space by first projecting them to 2d coordinates, and then clustering them using clustering algorithms. 
04. Feature representation
... Instead of only operating in pixel space, we can use more advanced teqniques to find better representations for each slice/patch. These will capture more important/promenant features. Among such methods are CPC, encoders and seismic attributes.  
05. Defining target structures using clustering technicques.
... Until now we have only looked at similarity based on arbitrary characteristics. Most often, one is interested in spesific target structures, and therefore we want to use the methodology from the previous chapters enable this. 
