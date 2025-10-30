def simulate_operation(n, permutation, k):
    """Simulate the operation at position k."""
    new_perm = permutation.copy()
    
    # Adjust for 0-indexing
    k_idx = k - 1
    
    # Sort the left part (before position k)
    if k >= 2:
        left_part = sorted(new_perm[:k_idx])
        for i in range(len(left_part)):
            new_perm[i] = left_part[i]
    
    # Sort the right part (after position k)
    if k <= n - 1:
        right_part = sorted(new_perm[k_idx+1:])
        for i in range(len(right_part)):
            new_perm[k_idx + 1 + i] = right_part[i]
    
    return new_perm

def solve(n, permutation):
    """Find the minimum number of operations required."""
    # Check if the permutation is already the identity permutation
    correct = sum(1 for i in range(n) if permutation[i] == i + 1)
    if correct == n:
        return 0

    operations = 0
    while correct < n:
        # Try each position k
        max_correct = -1
        best_perm = None
        for k in range(1, n + 1):
            # Simulate the operation at position k
            new_perm = simulate_operation(n, permutation, k)
            # Count how many elements are in the correct position
            new_correct = sum(1 for i in range(n) if new_perm[i] == i + 1)
            if new_correct > max_correct:
                max_correct = new_correct
                best_perm = new_perm
                # Speedup: If the permutation is already the identity, no need to check other positions
                if new_correct == n:
                    break

        # Apply the operation at the best position
        permutation = best_perm
        operations += 1
        correct = max_correct

    return operations

# Main function to process input and output
t = int(input())
for _ in range(t):
    n = int(input())
    permutation = list(map(int, input().split()))
    print(solve(n, permutation))