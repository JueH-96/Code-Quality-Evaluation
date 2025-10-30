# YOUR CODE HERE
def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Precompute prefix distinct counts
    prefix_distinct = [0] * (N + 1)
    seen = set()
    for i in range(N):
        seen.add(A[i])
        prefix_distinct[i + 1] = len(seen)
    
    # Precompute suffix distinct counts
    suffix_distinct = [0] * (N + 1)
    seen = set()
    for i in range(N - 1, -1, -1):
        seen.add(A[i])
        suffix_distinct[i] = len(seen)
    
    max_sum = 0
    
    for i in range(1, N):
        middle_seen = set()
        # Add the first element of the middle subarray
        middle_seen.add(A[i])
        
        for j in range(i + 1, N):
            # At this point, middle_seen has all elements from A[i] to A[j-1]
            
            distinct_subarray1 = prefix_distinct[i]
            distinct_subarray2 = len(middle_seen)
            distinct_subarray3 = suffix_distinct[j]
            
            current_sum = distinct_subarray1 + distinct_subarray2 + distinct_subarray3
            max_sum = max(max_sum, current_sum)
            
            # Add the next element to the middle_seen for the next iteration
            if j < N - 1:
                middle_seen.add(A[j])
    
    return max_sum

print(solve())