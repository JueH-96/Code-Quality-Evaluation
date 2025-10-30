def find_shortest_subarray_with_repeat(N, A):
    min_length = float('inf')
    
    # For each starting position
    for i in range(N):
        # Keep track of seen numbers
        seen = set()
        # For each ending position
        for j in range(i, N):
            # If we find a number we've seen before in current subarray
            if A[j] in seen:
                # Update minimum length if this subarray is shorter
                min_length = min(min_length, j - i + 1)
                # We can break inner loop as any longer subarray will be larger
                break
            seen.add(A[j])
    
    return min_length if min_length != float('inf') else -1

# Read input
N = int(input())
A = list(map(int, input().split()))

# Get and print result
result = find_shortest_subarray_with_repeat(N, A)
print(result)