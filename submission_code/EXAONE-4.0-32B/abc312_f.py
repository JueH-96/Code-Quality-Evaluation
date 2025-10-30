import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    n = int(data[0])
    M = int(data[1])
    items = []
    idx = 2
    for i in range(n):
        t = int(data[idx])
        x = int(data[idx+1])
        idx += 2
        items.append((t, x))
    
    A = []
    B = []
    C = []
    
    for t, x in items:
        if t == 0:
            A.append(x)
        elif t == 1:
            B.append(x)
        else:
            C.append(x)
            
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)
    
    prefixA = [0]
    for num in A:
        prefixA.append(prefixA[-1] + num)
        
    prefixB = [0]
    for num in B:
        prefixB.append(prefixB[-1] + num)
        
    prefixC = [0]
    for num in C:
        prefixC.append(prefixC[-1] + num)
        
    n0 = len(A)
    n1 = len(B)
    n2 = len(C)
    
    ans = 0
    max_k = min(M, n1)
    
    for k in range(0, max_k + 1):
        lo, hi = 0, n2
        t_req = n2 + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if prefixC[mid] >= k:
                t_req = mid
                hi = mid - 1
            else:
                lo = mid + 1
                
        if t_req > n2:
            continue
            
        if k + t_req > M:
            continue
            
        m0 = M - k - t_req
        if m0 < 0:
            continue
            
        sA = prefixA[min(m0, n0)]
        sB = prefixB[k]
        candidate = sA + sB
        if candidate > ans:
            ans = candidate
            
    print(ans)

if __name__ == "__main__":
    main()