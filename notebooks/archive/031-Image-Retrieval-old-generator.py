# now we create a generator to fetch similar images with a treshold
# if the image is not inside the same cluster, we go to the next cluster in the sorted_by_distance order. 

def euclidean(coords):
    xx, yy,_,_ = ref
    x, y,_,_ = coords
    return ((x-xx)**2 + (y-yy)**2)**0.5

    
class SimilarityImageGenerator(object):

    ''' Data generator providing a batch of  '''

    def __init__(self, ref, direction, treshold, unsorted):
    
        
        # Set params
        self.indx = 0
        self.ref = ref
        self.direction = direction
        self.treshold = treshold
        self.unsorted = unsorted
        self.ordered = self.order_by_dist(self.unsorted)#self.order_by_cluster_first(self.unsorted) 
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self.ordered)

    def next(self):
        if self.indx < len(self.ordered):
            # calculate distance in slices to the next in the ordered list
            slice_dist = abs(ref[-1] - self.ordered[self.indx][-1])
            #print(slice_dist)
            # continue to iterate until e find the closest one outside the threshold
            while  slice_dist < self.treshold:
                self.indx = self.indx+1
                slice_dist = abs(ref[-1] - self.ordered[self.indx][-1])
            self.indx = self.indx+1
            # return the most similar item outside the function 
            return self.ordered[self.indx]
        else:
          raise StopIteration()
        
    def order_by_cluster_first(self, unsorted): 
        # we allocate a data variable to store our ordered data
        # this will be a list ordered first by cluster, then by eucledian distance 
        ordered = []
        # then we loop through each cluster and yeld the first image that is closes to the ref, but maintainig minLisceIndex away 
        for i in cluster_order[dir]:
            inter_cluster_points =unsorted[unsorted[:,2] == i, :]
            # then we sort these inter-cluster points withing the current cluster
            inter_cluster_points = inter_cluster_points.tolist()
            #print(inter_cluster_points)
            print(len(inter_cluster_points))
            print("")
            inter_cluster_points.sort(key=euclidean) # 3 column wsorted on ecleduan dist from ref (x, y, slice index)
            ordered = ordered + inter_cluster_points
        #print(len(ordered_by_cluster_and_dist))
        #print(ordered_by_cluster_and_dist)
        return ordered
    
    def order_by_dist(self, unsorted): 
        ordered = unsorted.tolist()
        ordered.sort(key=euclidean)
        return ordered

    

          
        
  
direction = "xline"
treshold = 10
# first we group togheter the pca and the slice index labels
pca_cluster_labels = k_means[direction].labels_
pca_index_labels =  np.column_stack((pca[direction], pca_cluster_labels))
pca_index_labels =  np.column_stack((pca_index_labels, labels[direction])) 

ref = pca_index_labels[300]
print("REF", ref)
print("direction", direction)
print("treshold", treshold)

#print(pca[dir][k_means[dir].labels_ == 0, :])
ag = SimilarityImageGenerator(ref, direction, treshold, pca_index_labels)
plt.imshow(data[direction][int(ref[-1])].reshape(shape[direction][0], shape[direction][1]).T)
plt.title(f'Referance image at index {ref[-1]}')

plt.show()
i = 0 
for img in ag:
    print("most similar img", img)
    #plot_sequences(x, y, labels, output_path=r'resources/batch_sample_sorted.png')
    plt.imshow(data[direction][int(img[-1])].reshape(shape[direction][0], shape[direction][1]).T)
    plt.title(f'{i}th most similar image at index {img[-1]}')

    plt.show()
    i= i+1
    if i > 10:
        break

    