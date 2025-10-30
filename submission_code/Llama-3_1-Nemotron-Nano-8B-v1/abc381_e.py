import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    S = data[ptr]
    ptr += 1
    
    # Precompute prefix sums for '1's and '2's
    prefix_ones = [0] * (N + 1)
    prefix_twos = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_ones[i] = prefix_ones[i-1] + (S[i-1] == '1')
        prefix_twos[i] = prefix_twos[i-1] + (S[i-1] == '2')
    
    # Collect positions of '/' (1-based)
    div_positions = []
    for i in range(N):
        if S[i] == '/':
            div_positions.append(i + 1)  # Store as 1-based
    
    # Process each query
    for _ in range(Q):
        L = int(data[ptr])
        ptr += 1
        R = int(data[ptr])
        ptr += 1
        
        # Find the range of '/' in [L, R]
        left_idx = bisect.bisect_left(div_positions, L)
        right_idx = bisect.bisect_right(div_positions, R) - 1
        
        if left_idx > right_idx:
            print(0)
            continue
        
        sublist = div_positions[left_idx : right_idx + 1]
        best_m = 0
        low = 0
        high = len(sublist) - 1
        
        while low <= high:
            if low == high:
                i = sublist[low]
                a = prefix_ones[i-1] - prefix_ones[L-1]
                b = prefix_twos[R] - prefix_twos[i]
                current_m = min(a, b)
                best_m = max(best_m, current_m)
                break
            
            mid1 = low + (high - low) // 3
            mid2 = high - (high - low) // 3
            i1 = sublist[mid1]
            a1 = prefix_ones[i1-1] - prefix_ones[L-1]
            b1 = prefix_twos[R] - prefix_twos[i1]
            m1 = min(a1, b1)
            
            i2 = sublist[mid2]
            a2 = prefix_ones[i2-1] - prefix_ones[L-1]
            b2 = prefix_twos[R] - prefix_twos[i2]
            m2 = min(a2, b2)
            
            if m1 < m2:
                best_m = max(best_m, m2)
                low = mid1 + 1
            else:
                best_m = max(best_m, m1)
                high = mid2 - 1
        
        print(2 * best_m + 1)

if __name__ == '__main__':
    main()