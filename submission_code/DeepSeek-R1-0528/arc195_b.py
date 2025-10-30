import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    B = list(map(int, data[1+n:1+2*n]))
    
    fixed_sum = None
    for i in range(n):
        if A[i] != -1 and B[i] != -1:
            s = A[i] + B[i]
            if fixed_sum is None:
                fixed_sum = s
            elif fixed_sum != s:
                print('No')
                return
                
    if fixed_sum is not None and fixed_sum < 0:
        print('No')
        return
        
    if fixed_sum is None:
        M = -10**18
        for a in A:
            if a != -1:
                if a > M:
                    M = a
        for b in B:
            if b != -1:
                if b > M:
                    M = b
        if M < 0:
            S = 0
        else:
            S = M
    else:
        S = fixed_sum
        
    for a in A:
        if a != -1:
            if a > S:
                print('No')
                return
                
    for b in B:
        if b != -1:
            if b > S:
                print('No')
                return
                
    U0 = [a for a in A if a != -1]
    V0 = []
    for b in B:
        if b != -1:
            V0.append(S - b)
            
    countU = defaultdict(int)
    for a in U0:
        countU[a] += 1
        
    countV = defaultdict(int)
    for v in V0:
        countV[v] += 1
        
    keys = set(countU.keys()) | set(countV.keys())
    total = 0
    for k in keys:
        total += max(countU.get(k, 0), countV.get(k, 0))
        
    if total <= n:
        print('Yes')
    else:
        print('No')

if __name__ == "__main__":
    main()