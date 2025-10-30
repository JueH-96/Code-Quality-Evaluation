def main():
    import sys
    input_data = sys.stdin.buffer.read().split()
    if not input_data:
        return
    
    # Read N and M. The rest of the input (the M edges)
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # Even though we are given M edges, we do not actually need to process them.
    # The given graph is guaranteed to have no odd cycle so it is bipartite.
    # In any bipartite graph on n vertices, the maximum number of edges
    # is achieved by a complete bipartite graph. It is known that if we partition
    # the vertex set into two parts A and B, the number of edges is |A| * |B|.
    # The maximum value of |A|*|B| for A ∪ B = {1, ..., n} is attained when the parts
    # are as balanced as possible, i.e., |A| = floor(n/2) and |B| = ceil(n/2).
    #
    # So eventually the board can have at most 
    #    max_edges = floor(n/2) * ceil(n/2)
    # edges, and since the initial graph already has m edges, the number of edges that
    # can still be added legally is
    #    moves_left = max_edges - m.
    #
    # The move in the game is: add any missing edge that does not create an odd cycle.
    # (In a bipartite graph an edge is safe if and only if it connects vertices from the two parts.)
    #
    # Thus, no matter how the players choose their moves, they are essentially
    # picking from a set of "moves" of fixed size moves_left.
    # In a game where the players take alternating turns removing one object from a pile,
    # the first player wins if and only if the total number is odd.
    # Here Aoki (the first player) wins if moves_left is odd; otherwise, Takahashi wins.
    
    # Skip through the M edges (they are irrelevant for the counting)
    for _ in range(m):
        next(it)
        next(it)
    
    a = n // 2
    b = n - a  # which is ceil(n/2)
    max_edges = a * b
    moves_left = max_edges - m
    
    if moves_left % 2 == 1:
        sys.stdout.write("Aoki")
    else:
        sys.stdout.write("Takahashi")

if __name__ == '__main__':
    main()