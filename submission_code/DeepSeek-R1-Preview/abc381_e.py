import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    S = data[idx]
    idx += 1
    
    # Precompute prefix sums for '1's and '2's, 1-based
    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    slashes = []
    for i in range(1, N + 1):
        prefix_1[i] = prefix_1[i - 1] + (1 if S[i - 1] == '1' else 0)
        prefix_2[i] = prefix_2[i - 1] + (1 if S[i - 1] == '2' else 0)
        if S[i - 1] == '/':
            slashes.append(i)
    
    # Process each query
    for _ in range(Q):
        L = int(data[idx])
        idx += 1
        R = int(data[idx])
        idx += 1
        
        # Find slashes in [L, R]
        low = bisect.bisect_left(slashes, L)
        high = bisect.bisect_right(slashes, R)
        current_slashes = slashes[low:high]
        
        if not current_slashes:
            print(0)
            continue
        
        # Collect candidates: first, last, mid, mid-1, mid+1
        candidates = []
        candidates.append(current_slashes[0])
        candidates.append(current_slashes[-1])
        mid = len(current_slashes) // 2
        candidates.append(current_slashes[mid])
        if mid > 0:
            candidates.append(current_slashes[mid - 1])
        if mid < len(current_slashes) - 1:
            candidates.append(current_slashes[mid + 1])
        
        # Add more candidates if available
        if len(current_slashes) > 5:
            quarter = len(current_slashes) // 4
            candidates.append(current_slashes[quarter])
            candidates.append(current_slashes[-quarter])
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        max_len = 0
        for j in candidates:
            a = prefix_1[j - 1] - prefix_1[L - 1]
            b_val = prefix_2[R] - prefix_2[j]
            current_min = min(a, b_val)
            current_len = 2 * current_min + 1
            if current_len > max_len:
                max_len = current_len
        
        print(max_len)

if __name__ == '__main__':
    main()