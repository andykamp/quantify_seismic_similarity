import numpy as np
import tensorflow as tf
import tqdm
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib
#matplotlib.use('PS')
#matplotlib.use(
#    "TkAgg"
#)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
"""
see https://github.com/hrzn/jacobianmatrix/blob/master/Jacobian-matrix-examples.ipynb
from https://github.com/hrzn/jacobianmatrix/blob/master/Jacobian-matrix-examples.ipynb
along the lines of 
Fast Approximate Geodesics for Deep Generative Models
Nutan Chen, Francesco Ferroni, Alexej Klushyn, Alexandros Paraschos, Justin Bayer, Patrick van der Smagt
"""

class RiemannianMetric(object):
    def __init__(self, x, z, session):
        self.x = x
        self.z = z
        self.session = session
        print("nklnlnnkl", x, z)
        
    def create_tf_graph(self, output_dim):
        """
        creates the metric tensor (J^T J and J being the jacobian of the decoder), 
        which can be evaluated at any point in Z
        and
        the magnification factor
        """
        print("dsfsdsdfds", self.x[:, 0], self.z)
        # the metric tensor
        if not output_dim:
            output_dim = self.x.shape[1].value
        # derivative of each output dim wrt to input (tf.gradients would sum over the output)
        J = [tf.gradients(self.x[:, _], self.z)[0] for _ in range(output_dim)] # TODO HARDCOD RMEOOOOOOOOOVE
        J = tf.stack(J, axis=1)  # batch x output x latent
        self.J = J

        G = tf.transpose(J, [0, 2, 1]) @ J  # J^T \cdot J
        self.G = G

        # magnification factor meaninf sqrt(det(Reiammn metric )) THIS IS WHAT WE WANT TO PLOT
        MF = tf.sqrt(tf.linalg.det(G))
        self.MF = MF

    def riemannian_distance_along_line(self, z1, z2, n_steps):
        """
        calculates the riemannian distance between two near points in latent space on a straight line
        the formula is L(z1, z2) = \int_0^1 dt \sqrt(\dot \gamma^T J^T J \dot gamma)
        since gamma is a straight line \gamma(t) = t z_1 + (1-t) z_2, we get
        L(z1, z2) = \int_0^1 dt \sqrt([z_1 - z2]^T J^T J [z1-z2])
        L(z1, z2) = \int_0^1 dt \sqrt([z_1 - z2]^T G [z1-z2])
        z1: starting point
        z2: end point
        n_steps: number of discretization steps of the integral
        """

        # discretize the integral aling the line
        t = np.linspace(0, 1, n_steps)
        dt = t[1] - t[0]
        the_line = np.concatenate([_ * z1 + (1 - _) * z2 for _ in t])

        if True:
            # for weird reasons it seems to be alot faster to first eval G then do matrix mutliple outside of TF
            G_eval = self.session.run(self.G, feed_dict={self.z: the_line})

            # eval the integral at discrete point
            L_discrete = np.sqrt((z1-z2) @ G_eval @ (z1-z2).T)

            print("m", L_discrete)
            L_discrete = L_discrete.flatten()


            L = np.sum(dt * L_discrete)

        else:
            # THIS IS ALOT (10x) slower, although its all in TF
            DZ = tf.constant(z1 - z2)
            DZT = tf.constant((z1 - z2).T)
            tmp_ = tf.tensordot(self.G, DZT, axes=1)
            tmp_ = tf.einsum('j,ijk->ik', DZ[0], tmp_ )
            # tmp_ = tf.tensordot(DZ, tmp_, axes=1)

            L_discrete = tf.sqrt(tmp_)  # this is a function of z, since G(z)

            L_eval = self.session.run(L_discrete, feed_dict={self.z: the_line})
            L_eval = L_eval.flatten()
            L = np.sum(dt * L_eval)

        return L


class RiemannianTree(object):
    """docstring for RiemannianTree"""

    def __init__(self, riemann_metric):
        super(RiemannianTree, self).__init__()
        self.riemann_metric = riemann_metric  # decoder input (tf_variable)


    def create_riemannian_graph(self, z, n_steps, n_neighbors):

        n_data = len(z)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(z)

        G = nx.Graph()

        # Nodes
        for i in range(n_data):
            n_attr = {f'z{k}': float(z[i, k]) for k in range(z.shape[1])}
            G.add_node(i, **n_attr)

        # edges
        for i in tqdm.trange(n_data):
            distances, indices = knn.kneighbors(z[i:i+1])
            # first dim is for samples (z), but we only have one
            distances = distances[0]
            indices = indices[0]

            for ix, dist in zip(indices, distances):
                # calculate the riemannian distance of z[i] and its nn

                # save some computation if we alrdy calculated the other direction
                if (i, ix) in G.edges or (ix, i) in G.edges or i == ix:
                    continue

                L_riemann = self.riemann_metric.riemannian_distance_along_line(z[i:i+1], z[ix:ix+1], n_steps=n_steps)
                L_euclidean = dist

                # note nn-distances are NOT symmetric
                edge_attr = {'distance_riemann': float(1/L_riemann),
                             'weight_euclidean': float(1/L_euclidean),
                             'distance_riemann': float(L_riemann),
                             'distance_euclidean': float(L_euclidean)}
                G.add_edge(i, ix, **edge_attr)
        return G



def main():

    """
    IMPORTANT: 
    We calculate the rieman metric as the change from input to ouput. Hence it is how much the input space need to chenge in the process. 
    For an encoder, we can therefore visualize how much the input variables will change.
    But more interestingly, for a decoder (in VAE), we can visualize how much it changes from latent space to the original space
    This means you can stand in latent space, and know how much distorion it is around you. If alot of distorition(riemann magnificaiton is high),
    it means that you most likely look very different in input space x. therefor you should actually be far appart!

    """
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Input
    input_dim = 2
    latent_dim = 2
    m = Sequential()
    m.add(Dense(200, activation='tanh', input_shape=(input_dim, )))
    m.add(Dense(100, activation='tanh', ))
    m.add(Dense(latent_dim, activation='tanh'))

    m.summary()

    print("inputoutput", m.output, m.input, m.output.shape[1].value)
    print("nkln", m.output[:, 0], m.input)
    


    # plot the model real quick
    inp = np.random.uniform(-50,50, size=(1000, latent_dim))
    outp = m.predict(inp)

    plt.figure()
    plt.scatter(inp[:,0], inp[:,1])
    plt.figure()
    plt.scatter(outp[:,0], outp[:,1])


    session = tf.Session()
    session.run(tf.global_variables_initializer())

    rmetric = RiemannianMetric(x=m.output, z=m.input, session=session)
    rmetric.create_tf_graph()

    mf = session.run(rmetric.MF, {rmetric.z: inp})
    plt.figure()
    plt.scatter(inp[:,0], inp[:,1], c=mf)

    z1 = np.array([[1, 10]])
    z2 = np.array([[10, 2]])
    # for steps in [100,1_000,10_000,100_000]:
    #     q = r.riemannian_distance_along_line(z1, z2, n_steps=steps)
    #     print(q)
    plt.show()
    
    # try this on the swizz_roll dataset
    
    n_samples = 1000
    noise = 0.5
    import sklearn.datasets
    #z, _  = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=noise, random_state=None)
    z, _= sklearn.datasets.make_moons(n_samples=n_samples, noise=.05)
    # Make it thinner
    #z[:, 1] *= .5


    # visualize it in 3d 
    def viz_3D_swizz(): 
        import mpl_toolkits.mplot3d.axes3d as p3
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        ax.scatter(z[:, 0], z[:, 1], z[:, 2],
                s=20, edgecolor='k')
        plt.title('Swizz roll 3D')
        plt.show()

    def swizz3d_TO_2D():
        #Visualize a projection to 2d 
        # we simulate latency space by projecting the 3d swizz roll in 2d
        z = z[:,[0,2]]
        
    
    plt.scatter(z[:,0], z[:,1])
    plt.show()
    # plot the model real quick
    inp = z
    outp = m.predict(z)

    

    mf = session.run(rmetric.MF, {rmetric.z: inp})
    print("mf swiss", mf.shape)
    plt.figure()
    plt.scatter(inp[:,0], inp[:,1], c=mf) # how much the different points are going to change
    plt.scatter(outp[:,0], outp[:,1], c="red")
    plt.colorbar()
    plt.show()

  
    # plot mf for entire grid 
    h = .09  # step size in the mesh
    x_min, x_max = inp[:, 0].min() - 1, inp[:, 0].max() + 1
    y_min, y_max = inp[:, 1].min() - 1,inp[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    grid_inp = np.c_[xx.ravel(), yy.ravel()]   
    mf = session.run(rmetric.MF, {rmetric.z: grid_inp})
    print("mf grid", mf.shape)
    print(xx.shape, yy.shape, mf.shape)
    # Put the result into a color plot
    Z = mf.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap="brg")

    #plot actuals
    plt.scatter(inp[:,0], inp[:,1],  )
    plt.scatter(outp[:,0], outp[:,1],  )
    plt.title("Full mf showing how inp space will change ")
    plt.colorbar()
    plt.show()


    # TODO NOT USED
    #outp = m.predict(z)
    #plt.figure()
    #plt.scatter(outp[:,0], outp[:,1])
    #plt.show()

    # we initi the riemanntree 
    rTree = RiemannianTree(rmetric)
    # we then calculate the eucledian and riemann/geodesic length for each edge 
    G = rTree.create_riemannian_graph(z, n_steps=1000, n_neighbors=10)

    # can use G to do shortest path finding now



    print("number_of_nodes", G.number_of_nodes())
    print("number_of_edges", G.number_of_edges())
    #print("nodes", G.nodes())
    #print("edges", G.edges())
    
    # plotting each graph
    q = rmetric.riemannian_distance_along_line(np.array([[-10, 0]]), np.array([[5, 0]]), n_steps=100)
    print(q)

    shortest_path_eucl = nx.algorithms.shortest_paths.generic.shortest_path(G, 0, 50, "weight_euclidean")
    shortest_path_rie = nx.algorithms.shortest_paths.generic.shortest_path(G, 0, 50, "weight")

    def plotPath(data, pathList, G):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        n = len(pathList)
        colors = plt.cm.jet(np.linspace(0,1,n))
        i = 0 
        for path in pathList:
            print("path", path)
            shortest_weight_eucl = 0
            shortest_weight_rie = 0
            # find
            for indx in range(len(path)-1):
                fr = path[indx]
                to = path[indx+1]
                edge = G[fr][to]
                #print(fr, to, edge)
                shortest_weight_eucl += edge["distance_euclidean"]
                shortest_weight_rie += edge["distance_riemann"]
            print(path, shortest_weight_eucl, shortest_weight_rie)
            
           
            # plot line trough shortest path
            ax1.plot(data[path,0], data[path,1], color=colors[i], label='Line ' + str(i)  )
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels)
            
            i+=1
        # plot all nodes
        ax1.scatter(data[:,0], data[:,1])
        plt.show()
    
    pathList = [shortest_path_eucl, shortest_path_rie]
    plotPath(z, pathList, G)



    nx.draw(G)
    plt.show()
   
if __name__ == '__main__':
    main()
