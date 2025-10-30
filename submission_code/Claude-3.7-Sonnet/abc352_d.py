def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))
    
    # Store the 1-indexed position of each value in the permutation
    positions = [0] * (n + 1)  # positions[v] is the position of value v in the permutation
    for i, val in enumerate(p):
        positions[val] = i + 1  # Convert to 1-indexed
    
    min_diff = float('inf')
    # Try all possible consecutive sequences of K integers
    for a in range(1, n - k + 2):  # 'a' is the start of a consecutive range
        # Get the positions of all values in the range [a, a+k-1]
        indices = [positions[v] for v in range(a, a + k)]
        # Calculate the span (i_K - i_1) and update minimum
        min_diff = min(min_diff, max(indices) - min(indices))
    
    return min_diff

print(solve())