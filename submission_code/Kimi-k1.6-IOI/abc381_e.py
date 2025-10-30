import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    q = int(data[idx+1])
    idx +=2
    s = data[idx]
    idx +=1
    
    pre1 = [0]*(n+1)
    pre2 = [0]*(n+1)
    for i in range(n):
        pre1[i+1] = pre1[i] + (1 if s[i] == '1' else 0)
        pre2[i+1] = pre2[i] + (1 if s[i] == '2' else 0)
    
    slashes = [i for i, c in enumerate(s) if c == '/']
    m = len(slashes)
    
    for _ in range(q):
        L = int(data[idx])
        R = int(data[idx+1])
        idx +=2
        L0 = L-1
        R0 = R-1
        
        # Check if there's any slash in [L0, R0]
        if m ==0:
            print(0)
            continue
        slash_idx = bisect.bisect_left(slashes, L0)
        has_slash = False
        if slash_idx < m and slashes[slash_idx] <= R0:
            has_slash = True
        if not has_slash:
            print(0)
            continue
        
        total_1s = pre1[R0+1] - pre1[L0]
        total_2s = pre2[R0+1] - pre2[L0]
        max_k = min(total_1s, total_2s)
        
        low = 0
        high = max_k
        ans = 0
        
        while low <= high:
            mid = (low + high) //2
            A = pre1[L0] + mid
            B = pre2[R0+1] - mid
            
            p_min = bisect.bisect_left(pre1, A)
            upper_p_bisect = bisect.bisect_right(pre2, B)
            upper_p = upper_p_bisect -2
            
            a = max(L0, p_min)
            b = min(R0, upper_p)
            
            feasible = False
            if a <= b:
                pos = bisect.bisect_left(slashes, a)
                if pos < m and slashes[pos] <= b:
                    feasible = True
            
            if feasible:
                ans = mid
                low = mid +1
            else:
                high = mid -1
        
        print(2*ans +1)

if __name__ == "__main__":
    main()