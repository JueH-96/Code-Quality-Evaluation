def max_distinct_counts(N, A):
    # Compute distinct_prefix
    distinct_prefix = [0]
    seen_prefix = set()
    for num in A:
        seen_prefix.add(num)
        distinct_prefix.append(len(seen_prefix))
    
    # Compute distinct_suffix
    distinct_suffix = [0] * (N + 1)
    seen_suffix = set()
    for i in range(N-1, -1, -1):
        seen_suffix.add(A[i])
        distinct_suffix[i] = len(seen_suffix)
    
    # Find the maximum sum
    max_sum = 0
    for i in range(1, N):
        current_sum = distinct_prefix[i] + distinct_suffix[i]
        max_sum = max(max_sum, current_sum)
    
    return max_sum

N = int(input().strip())
A = list(map(int, input().strip().split()))

print(max_distinct_counts(N, A))