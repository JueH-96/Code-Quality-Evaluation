import sys

def solve():
    """
    This function reads the input, solves the strongest programmer problem,
    and prints the output to standard output.
    """
    # Read N (number of programmers) and M (number of information pieces)
    try:
        line = sys.stdin.readline()
        if not line:
            return
        N, M = map(int, line.split())
    except (ValueError, IndexError):
        # Exit gracefully on malformed input, not expected under problem constraints
        return

    # `reachable[i][j]` will be True if we can infer that person i+1 is
    # stronger than person j+1. We use 0-based indexing for the matrix.
    reachable = [[False] * N for _ in range(N)]

    # Read the M relations and populate the initial reachability matrix.
    for _ in range(M):
        try:
            line = sys.stdin.readline()
            if not line:
                break
            A, B = map(int, line.split())
            # Convert 1-based person numbers to 0-based indices.
            u, v = A - 1, B - 1
            reachable[u][v] = True
        except (ValueError, IndexError):
            break

    # Use the Floyd-Warshall algorithm to compute the transitive closure.
    # After this loop, reachable[i][j] is True if there is a path from i to j.
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if reachable[i][k] and reachable[k][j]:
                    reachable[i][j] = True

    # Find all candidates for the strongest programmer.
    # A person 'p' is a candidate if no other person 'q' is known to be
    # stronger, meaning there's no path from 'q' to 'p'.
    candidates = []
    for p_idx in range(N):
        is_beaten = False
        # Check if any other person q is known to be stronger than p.
        for q_idx in range(N):
            if reachable[q_idx][p_idx]:
                is_beaten = True
                break
        
        if not is_beaten:
            # If no one is known to be stronger, this person is a candidate.
            # We store their original 1-based number.
            candidates.append(p_idx + 1)

    # If there is exactly one candidate, we have a unique answer.
    # Otherwise, it's ambiguous.
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

# Execute the solution
solve()