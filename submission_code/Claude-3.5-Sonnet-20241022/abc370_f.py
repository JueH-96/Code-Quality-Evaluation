def check_division(A, N, K, x):
    # Try to divide the cake into K parts where each part has sum >= x
    cnt = 0
    curr_sum = 0
    for i in range(2*N):
        curr_sum += A[i % N]
        if curr_sum >= x:
            cnt += 1
            curr_sum = 0
            if cnt >= K:
                return True
    return False

def get_min_sum(A, N, K):
    # Binary search for the maximum possible minimum sum
    left = 1
    right = sum(A)
    
    while left < right:
        mid = (left + right + 1) // 2
        if check_division(A, N, K, mid):
            left = mid
        else:
            right = mid - 1
            
    return left

def count_uncut_lines(A, N, K, min_sum):
    # For each cut line, check if it must always be uncut
    uncut = [1] * N
    
    # Try all possible starting positions
    for start in range(N):
        curr_sum = 0
        parts = []
        curr_part = []
        
        # Try to make a valid division starting from this position
        for i in range(N):
            idx = (start + i) % N
            curr_sum += A[idx]
            curr_part.append(idx)
            
            if curr_sum >= min_sum:
                parts.append(curr_part)
                curr_sum = 0
                curr_part = []
                
                if len(parts) == K:
                    # Valid division found
                    # Mark cut lines that are cut in this division
                    for part in parts:
                        for i in range(len(part)-1):
                            uncut[part[i]] = 0
                        if len(part) > 0:
                            uncut[(part[-1] + N - 1) % N] = 0
                    break
    
    return sum(uncut)

N, K = map(int, input().split())
A = list(map(int, input().split()))

min_sum = get_min_sum(A, N, K)
uncut_count = count_uncut_lines(A, N, K, min_sum)

print(min_sum, uncut_count)