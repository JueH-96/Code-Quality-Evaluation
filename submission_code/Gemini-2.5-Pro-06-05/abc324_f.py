import sys

def main():
    """
    Main function to solve the problem.
    It reads the graph, performs binary search on the ratio,
    and prints the maximum possible ratio.
    """
    # Use sys.stdin.readline for faster I/O with large inputs.
    input = sys.stdin.readline

    # Read the number of vertices and edges.
    try:
        N, M = map(int, input().split())
    except (ValueError, IndexError):
        # Handles empty input lines if they occur.
        return

    # Represent the graph using an adjacency list.
    # adj[u] stores a list of tuples (v, beauty, cost) for edges from u.
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v, b, c = map(int, input().split())
        adj[u].append((v, b, c))

    def check(x: float) -> bool:
        """
        Check if a ratio of at least x is achievable.
        This is done by finding the longest path from 1 to N with
        edge weights redefined as (beauty - x * cost).
        """
        dp = [-float('inf')] * (N + 1)
        dp[1] = 0.0

        # The constraint u_i < v_i ensures the graph is a DAG and
        # vertices are already topologically sorted.
        for u in range(1, N + 1):
            if dp[u] == -float('inf'):
                # Node u is not reachable from 1.
                continue
            
            # Relax outgoing edges from u.
            for v, b, c in adj[u]:
                weight = b - x * c
                if dp[v] < dp[u] + weight:
                    dp[v] = dp[u] + weight
        
        # If the longest path to N has a non-negative score, the ratio x is possible.
        return dp[N] >= 0

    # Binary search for the maximum ratio.
    low = 0.0
    high = 10001.0  # A safe upper bound for the ratio.

    # 100 iterations provide precision far greater than 10^-9.
    for _ in range(100):
        mid = low + (high - low) / 2.0
        if check(mid):
            # If ratio `mid` is possible, try for a higher one.
            low = mid
        else:
            # Otherwise, the answer must be less than `mid`.
            high = mid
            
    # Print the result with the required precision.
    print(f"{low:.15f}")

if __name__ == "__main__":
    main()