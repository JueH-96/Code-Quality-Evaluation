def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    
    queries = []
    idx = 3
    for _ in range(Q):
        L = int(data[idx]) - 1
        R = int(data[idx + 1]) - 1
        queries.append((L, R))
        idx += 2
    
    # Precompute prefix sums for '1', '/' and '2' counts
    prefix_ones = [0] * (N + 1)
    prefix_slashes = [0] * (N + 1)
    prefix_twos = [0] * (N + 1)
    
    for i in range(N):
        prefix_ones[i + 1] = prefix_ones[i] + (1 if S[i] == '1' else 0)
        prefix_slashes[i + 1] = prefix_slashes[i] + (1 if S[i] == '/' else 0)
        prefix_twos[i + 1] = prefix_twos[i] + (1 if S[i] == '2' else 0)
    
    results = []
    
    for L, R in queries:
        # Calculate the number of '1', '/' and '2' in the substring S[L:R+1]
        count_ones = prefix_ones[R + 1] - prefix_ones[L]
        count_slashes = prefix_slashes[R + 1] - prefix_slashes[L]
        count_twos = prefix_twos[R + 1] - prefix_twos[L]
        
        if count_slashes == 0:
            results.append(0)
        else:
            # We need at least one '/', and equal or more '1's and '2's on each side
            min_side = min(count_ones, count_twos)
            if min_side > 0:
                # The length of the largest valid 11/22 string we can form
                max_length = min(min_side, (count_slashes + 1) // 2) * 2 + 1
                results.append(max_length)
            else:
                results.append(0)
    
    for result in results:
        print(result)