In this example, we will learn how to model a shortest path problem as an OR problem. A shortest path problem is defined in the context of a
graph $G=(N,E)$, where $N$ is the set of nodes and $E$ is the set of directional arcs connecting nodes. In every shortest path, we have a commodity
moving from a source node $s \in N$ to a targte node $t \in N$ in a way that minimizes the overall distance transversed. This is the literal
definition of a shortest path, where the cost of each edge is its distance. In other applications, the cost could be related to time, an actual
monetary cost, or another combination of relevant factors. The definition of the cost does not change the way the shortest path is implemented.
In essence, we need to satisfy the following constraints:

- the commodity must leave the source node $s$
- the commodity must end its journey in the sink node $t$
- in every other node, conservation node must be enforced (if the commodity arrives to a node, then the commodity must leave the node as well)

In the example provided here, we must help a knight get back to his castle while moving as efficiently as possible in an enchanted forest, as shown below

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
