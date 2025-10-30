def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # Read the rest of the edges. Even though the input graph is bipartite and we are not
    # directly using the edges to compute the solution, we need to read them.
    edges = []
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        edges.append((u,v))
    
    # Explanation:
    #
    # The graph initially has no odd cycle so it is bipartite. A move is allowed if the added edge
    # does not create an odd cycle -- that is, it must connect one part of the bipartition to the other.
    #
    # Even if some bipartite components have a fixed bipartition (up to swap), in each component the players 
    # can flip the assignment (and assign isolated vertices arbitrarily) so that the overall bipartition of
    # vertices becomes as balanced as possible.
    #
    # In a bipartite graph with parts of sizes A and B, the complete bipartite graph has A * B edges.
    # Over all N vertices the maximum number of edges that can be added without creating an odd cycle
    # is achieved when the vertex set is split as evenly as possible.
    # Hence, the maximum possible number of edges in a bipartite graph on N vertices is:
    #    floor(N/2) * ceil(N/2)
    #
    # Let max_edges be that number.
    # Initially, M edges exist.
    # So the number of moves (ie, edges that can be added) is:
    #    moves = (floor(N/2)*ceil(N/2)) - M.
    #
    # The game is one where players alternate moves.
    # If the number of moves is odd, then the first mover wins (because after his final move, the opponent
    # cannot move). Otherwise the second mover wins.
    
    a = N // 2
    b = N - a
    max_edges = a * b
    moves = max_edges - M
    
    if moves % 2 == 1:
        sys.stdout.write("Aoki")
    else:
        sys.stdout.write("Takahashi")


if __name__ == '__main__':
    main()