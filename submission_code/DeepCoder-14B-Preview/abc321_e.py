def solve():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    results = []
    
    for _ in range(T):
        N = int(input[idx])
        X = int(input[idx + 1])
        K = int(input[idx + 2])
        idx += 3
        
        if K == 0:
            results.append(1)
            continue
        
        total = 0
        
        # Count nodes in the subtree of X at depth K
        if K <= 60:
            pow2k = 1 << K
            start_sub = X * pow2k
            if start_sub <= N:
                end_sub = start_sub + pow2k - 1
                end_sub = min(end_sub, N)
                total += end_sub - start_sub + 1
        
        # Count nodes in ancestor branches
        max_m = min(K, 60)
        for m in range(1, max_m + 1):
            A = X >> m
            if A == 0:
                break
            
            if m > K:
                continue
            n = K - m
            if n < 0:
                continue
            
            if n == 0:
                total += 1
            else:
                if (X == (A << 1)):
                    other_child = (A << 1) + 1
                else:
                    other_child = A << 1
                
                if other_child > N:
                    continue
                
                depth = n - 1
                if depth >= 60:
                    continue
                
                pow2depth = 1 << depth
                start_other = other_child * pow2depth
                if start_other > N:
                    continue
                
                end_other = start_other + pow2depth - 1
                end_other = min(end_other, N)
                cnt = end_other - start_other + 1
                total += cnt
        
        results.append(total)
    
    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == '__main__':
    solve()