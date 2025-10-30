def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    # We'll collect results to print at the end for efficiency.
    results = []
    
    for _ in range(t):
        N = int(data[idx]); idx += 1
        P = list(map(int, data[idx:idx+N]))
        idx += N
        
        # 0) Check if already sorted.
        if all(P[i] == i+1 for i in range(N)):
            results.append('0')
            continue
        
        # 1) Build pos array: pos[x] = the 1-based index of element x in P.
        pos = [0]*(N+1)
        for i in range(N):
            pos[P[i]] = i + 1
        
        # 2) Build prefixMax and suffixMin arrays.
        # prefixMax[i] = max(pos[1..i])
        prefixMax = [0]*(N+1)
        pm = 0
        for i in range(1, N+1):
            pm = max(pm, pos[i])
            prefixMax[i] = pm
        
        # suffixMin[i] = min(pos[i..N])
        suffixMin = [0]*(N+2)
        sm = N+1
        suffixMin[N+1] = sm
        for i in range(N, 0, -1):
            sm = min(sm, pos[i])
            suffixMin[i] = sm
        
        # 3) Check if one operation can fix it.
        #    We look for any k in [1..N] such that:
        #       pos[k] == k,
        #       prefixMax[k-1] < k,
        #       suffixMin[k+1] > k.
        #    If found, answer is 1; otherwise answer is 2.
        ans = 2
        for k in range(1, N+1):
            if pos[k] == k and prefixMax[k-1] < k and suffixMin[k+1] > k:
                ans = 1
                break
        
        results.append(str(ans))
    
    # Print all results.
    print('
'.join(results))

# Do not forget to call main().
if __name__ == '__main__':
    main()