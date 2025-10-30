import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    T = int(data[ptr])
    ptr += 1
    results = []
    
    for _ in range(T):
        N = int(data[ptr])
        ptr += 1
        P_list = list(map(int, data[ptr:ptr+N]))
        ptr += N
        
        # Convert P to 1-based index
        P = [0] * (N + 1)
        for i in range(N):
            P[i+1] = P_list[i]
        
        # Check if already sorted
        is_sorted = True
        for i in range(1, N+1):
            if P[i] != i:
                is_sorted = False
                break
        if is_sorted:
            results.append("0")
            continue
        
        # Precompute prefix_max and suffix_min
        prefix_max = [0] * (N + 2)
        for i in range(1, N+1):
            prefix_max[i] = max(prefix_max[i-1], P[i])
        
        suffix_min = [float('inf')] * (N + 2)
        for i in range(N, 0, -1):
            suffix_min[i] = min(P[i], suffix_min[i+1])
        
        found = False
        for k in range(1, N+1):
            if P[k] == k:
                # Check left_ok
                left_ok = True
                if k > 1:
                    if prefix_max[k-1] > k-1:
                        left_ok = False
                
                # Check right_ok
                right_ok = True
                if k < N:
                    if suffix_min[k+1] < k+1:
                        right_ok = False
                
                if left_ok and right_ok:
                    found = True
                    break
        
        if found:
            results.append("1")
        else:
            results.append("2")
    
    print('
'.join(results))

if __name__ == "__main__":
    main()