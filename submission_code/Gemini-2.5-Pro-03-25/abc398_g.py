import sys

# Set higher recursion depth if needed for graph algorithms, although this solution doesn't use recursion directly.
# sys.setrecursionlimit(200000) 

def solve():
    # Read the number of vertices N and the number of initial edges M from stdin
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the M edges to consume the input lines according to the specified format.
    # Based on the derived hypothesis for solving the problem, the specific details
    # of the edges (which vertices they connect) are not needed. We only need N and M.
    for _ in range(M):
        sys.stdin.readline()

    # The game involves adding edges to a bipartite graph such that it remains bipartite.
    # The game ends when no more edges can be added without violating the bipartite property
    # (i.e., without creating an odd cycle).
    # The final state of the graph must be a maximal bipartite graph.
    # A maximal bipartite graph on N vertices must be a connected graph and specifically
    # a complete bipartite graph K_{s,t} for some s, t such that s + t = N.
    
    # The number of edges in a complete bipartite graph K_{s,t} is s * t.
    # This value is maximized when s and t are as close as possible, i.e.,
    # s = floor(N/2) and t = ceil(N/2).
    # The maximum number of edges in any bipartite graph on N vertices is floor(N/2) * ceil(N/2).
    # This product is mathematically equivalent to floor(N^2 / 4).
    # We use integer division `//` which computes the floor value.
    max_edges_bipartite = (N * N) // 4
    
    # The core hypothesis is that regardless of the players' strategies, the game always ends
    # after a fixed number of moves, leading to a state with max_edges_bipartite edges.
    # The total number of moves K played in the game is the difference between the number
    # of edges in the final state and the number of edges in the initial state.
    # K = (Edges in final state) - (Edges in initial state)
    total_moves = max_edges_bipartite - M

    # Sanity check: The problem states the initial graph G is bipartite.
    # The number of edges M in a bipartite graph on N vertices cannot exceed max_edges_bipartite.
    # Therefore, M <= max_edges_bipartite, which guarantees that total_moves >= 0.
    
    # This is an impartial game. The outcome for optimal play is determined by the total number of moves.
    # If the total number of moves is odd, the first player (Aoki) wins.
    # If the total number of moves is even, the second player (Takahashi) wins.
    if total_moves % 2 == 1:
        # Odd number of moves implies the first player makes the last move.
        print("Aoki")
    else:
        # Even number of moves implies the second player makes the last move.
        # This correctly handles the case total_moves = 0:
        # If K = 0, Aoki has no valid moves at the start and loses immediately.
        # Since 0 is even, this branch correctly prints "Takahashi".
        print("Takahashi")

# Call the solve function to run the program
solve()