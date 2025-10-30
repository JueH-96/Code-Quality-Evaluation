import bisect

def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Compute prefix sums
    prefix = [0]
    for a in A:
        prefix.append(prefix[-1] + a)
    
    # Compute remainders
    remainders = [prefix[i] % M for i in range(N + 1)]
    
    # For each remainder, store the indices with that remainder, sorted
    indices = [[] for _ in range(M)]
    for i in range(N):  # t-1 ranges from 0 to N-1
        indices[remainders[i]].append(i)
    
    # Count pairs
    total_count = 0
    for s in range(1, N + 1):
        s_remainder = remainders[s - 1]
        
        # For t > s, count indices t-1 in indices[R[s-1]] such that t-1 >= s
        same_remainder_indices = indices[s_remainder]
        pos = bisect.bisect_left(same_remainder_indices, s)
        total_count += len(same_remainder_indices) - pos
        
        # For t < s, count indices t-1 in indices[(R[s-1] - R[N] + M) % M] such that t-1 <= s-2
        target_remainder = (s_remainder - remainders[N] + M) % M
        target_indices = indices[target_remainder]
        pos = bisect.bisect_right(target_indices, s - 2)
        total_count += pos
    
    return total_count

print(solve())