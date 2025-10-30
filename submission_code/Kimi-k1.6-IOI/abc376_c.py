def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N-1]))
    ptr += N-1
    
    A.sort()
    B.sort()
    m = N - 1
    
    # Compute max_left array
    max_left = [-float('inf')] * (m + 1)
    max_left[0] = -float('inf')
    for k in range(1, m + 1):
        current_diff = A[k-1] - B[k-1]
        max_left[k] = max(max_left[k-1], current_diff)
    
    # Compute suffix_max array
    suffix_max = [-float('inf')] * (m + 2)  # suffix_max[1] to suffix_max[m+1]
    suffix_max[m + 1] = -float('inf')
    for k in range(m, 0, -1):
        current_diff = A[k] - B[k-1]
        suffix_max[k] = max(current_diff, suffix_max[k+1])
    
    min_x = float('inf')
    
    for k in range(0, m + 1):
        if max_left[k] > 0:
            continue
        
        # Check right_ok
        if (k + 1 > m) or (suffix_max[k+1] <= 0):
            # Compute candidate_x
            if k == 0:
                if m == 0:
                    continue  # This case is impossible as N >=2
                if A[0] > B[0]:
                    continue
                candidate = A[0]
            elif k == m:
                candidate = max(A[m], B[m-1])
            else:
                lower = max(A[k], B[k-1])
                upper = B[k]
                if lower > upper:
                    continue
                candidate = lower
            
            if candidate < min_x:
                min_x = candidate
    
    if min_x == float('inf'):
        print(-1)
    else:
        print(min_x)

if __name__ == "__main__":
    main()