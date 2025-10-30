# YOUR CODE HERE
import sys

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Fast I/O
    input = sys.stdin.readline

    # Read N and M
    try:
        line = input()
        if not line:
            return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        # Handle empty input or malformed line
        return

    # Build adjacency list (0-indexed)
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)

    # Read weights W and initial piece counts A
    W = list(map(int, input().split()))
    A = list(map(int, input().split()))

    # dp[i] stores the maximum number of operations from a single piece on vertex i.
    dp = [0] * N

    # Process vertices in increasing order of weight. This establishes a valid
    # computational order for the DP, as dp[i] only depends on dp[j] where W[j] < W[i].
    sorted_vertices = sorted(range(N), key=lambda i: W[i])

    for i in sorted_vertices:
        # The choice of which new pieces to create from a piece on vertex `i`
        # is a 0/1 knapsack problem.
        # Capacity of the knapsack: W[i] - 1
        # Items: neighbors `j` with W[j] < W[i]
        # Item weight: W[j]
        # Item value: dp[j]

        capacity = W[i] - 1

        # We only consider neighbors with weights strictly less than W[i].
        items = []
        for neighbor in adj[i]:
            if W[neighbor] < W[i]:
                items.append((W[neighbor], dp[neighbor]))

        max_value_from_new_pieces = 0
        
        # The knapsack code handles capacity=0 correctly (result is 0).
        if items and capacity >= 0:
            knapsack_dp = [0] * (capacity + 1)
            
            for item_weight, item_value in items:
                # Iterate downwards to ensure each item is used at most once.
                for w in range(capacity, item_weight - 1, -1):
                    knapsack_dp[w] = max(knapsack_dp[w], knapsack_dp[w - item_weight] + item_value)
            
            max_value_from_new_pieces = knapsack_dp[capacity]
        
        # The total operations from one piece on vertex `i` is 1 (for the current operation)
        # plus the operations from the newly created pieces.
        dp[i] = 1 + max_value_from_new_pieces

    # The final answer is the sum of operations from all initial pieces.
    total_operations = 0
    for i in range(N):
        total_operations += A[i] * dp[i]
        
    print(total_operations)

if __name__ == "__main__":
    main()