import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    n = int(data[ptr])
    ptr += 1
    q = int(data[ptr])
    ptr += 1
    s = data[ptr]
    ptr += 1
    
    # Precompute prefix sums for '1's and '2's
    pre_one = [0] * (n + 1)
    pre_two = [0] * (n + 1)
    for i in range(1, n+1):
        pre_one[i] = pre_one[i-1] + (1 if s[i-1] == '1' else 0)
        pre_two[i] = pre_two[i-1] + (1 if s[i-1] == '2' else 0)
    
    # Collect positions of slashes (1-based)
    slashes = []
    for i in range(n):
        if s[i] == '/':
            slashes.append(i + 1)  # converting to 1-based index
    
    # Process each query
    results = []
    for _ in range(q):
        L = int(data[ptr])
        ptr += 1
        R = int(data[ptr])
        ptr += 1
        
        # Check if there are any slashes in [L, R]
        left = bisect.bisect_left(slashes, L)
        right_idx = bisect.bisect_right(slashes, R)
        if left >= right_idx:
            results.append(0)
            continue
        
        # Compute total 1's and 2's in [L, R]
        total_ones = pre_one[R] - pre_one[L-1]
        total_twos = pre_two[R] - pre_two[L-1]
        max_k = min(total_ones, total_twos)
        
        low = 0
        high = max_k
        ans = 0
        
        while low <= high:
            mid = (low + high) // 2
            
            # Calculate A and B for feasibility check
            A = pre_one[L-1] + mid
            idx = bisect.bisect_left(pre_one, A)
            lower_p = max(L, idx + 1)
            
            B = pre_two[R] - mid
            index = bisect.bisect_right(pre_two, B)
            upper_p = min(R, index - 1)
            
            # Check if there's any slash in [lower_p, upper_p]
            sl_left = bisect.bisect_left(slashes, lower_p)
            sl_right = bisect.bisect_right(slashes, upper_p)
            
            if sl_right > sl_left:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        
        results.append(2 * ans + 1)
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()