def compute_E(N, A, X, Y):
    memo = {0: 0.0}
    stack = [N]
    
    while stack:
        N = stack.pop()
        if N in memo:
            continue  # Already computed
        
        # Check if all dependencies are computed
        dependencies = [N // A] + [N // b for b in range(1, 7)]
        if all(d in memo for d in dependencies):
            cost1 = X + memo[N // A]
            cost2 = Y + sum(memo[N // b] for b in range(1, 7)) / 6
            memo[N] = min(cost1, cost2)
        else:
            # Not all dependencies are computed, push back to stack
            stack.append(N)
            # Push dependencies to stack if not already present
            for d in dependencies:
                if d not in memo:
                    stack.append(d)
    
    return memo[N]

# Read input
N, A, X, Y = map(int, input().split())

# Compute and print the result
print("{0:.12f}".format(compute_E(N, A, X, Y)))