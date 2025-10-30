def binomial(n, k):
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Take advantage of symmetry
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def multinomial(n, counts):
    result = 1
    remaining = n
    for count in counts:
        if count > 0:
            result *= binomial(remaining, count)
            remaining -= count
    return result

def solve(N, K):
    total_length = N * K
    
    # Calculate total number of sequences using multinomial coefficient
    S = multinomial(total_length, [K] * N)
    
    # Target rank (1-indexed)
    target = (S + 1) // 2
    
    # Current sequence
    sequence = []
    
    # Count of each number used so far
    used = [0] * (N + 1)  # 1-indexed
    
    for pos in range(total_length):
        remaining_length = total_length - pos
        
        # Try each possible number
        for num in range(1, N + 1):
            if used[num] < K:  # We can still use this number
                # Count sequences starting with this number
                new_used = used[:]
                new_used[num] += 1
                
                remaining_counts = []
                for i in range(1, N + 1):
                    remaining_counts.append(K - new_used[i])
                
                count = multinomial(remaining_length - 1, remaining_counts)
                
                if target <= count:
                    # Use this number
                    sequence.append(num)
                    used[num] += 1
                    break
                else:
                    # Skip this number
                    target -= count
    
    return sequence

N, K = map(int, input().split())
result = solve(N, K)
print(' '.join(map(str, result)))