import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # The problem describes an impartial game where players add edges to a bipartite graph.
    # The game ends when no more edges can be added without creating an odd cycle.
    # This means the graph eventually becomes a complete bipartite graph K_{S0, S1}.

    # For a graph with N vertices, the maximum number of edges a bipartite graph can have
    # is achieved when the two partitions, S0 and S1, are as close in size as possible.
    # This occurs when S0 = N // 2 and S1 = N - (N // 2).
    # The maximum number of edges is then (N // 2) * (N - N // 2).

    # The game rules state that players can choose any valid pair (i,j) to add an edge.
    # Because players can always choose how to connect disconnected components (by selecting
    # which partition-sides to connect), they can collectively ensure that the final
    # complete bipartite graph will be the one with the maximum possible edges (K_{N//2, N - N//2}).
    # This makes the total number of moves in the game a fixed value, regardless of specific choices.

    # Calculate the maximum possible edges in a bipartite graph on N vertices.
    s0 = N // 2
    s1 = N - s0
    max_bipartite_edges = s0 * s1

    # The total number of moves available in the game is the difference between this maximum
    # and the initial number of edges M.
    total_moves = max_bipartite_edges - M

    # In a normal play game, if the total number of moves is odd, the first player (Aoki) wins.
    # If the total number of moves is even, the second player (Takahashi) wins.
    if total_moves % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

solve()