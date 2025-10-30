import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    type0 = []
    type1 = []
    type2 = []
    
    for _ in range(N):
        T = int(input[ptr])
        X = int(input[ptr+1])
        ptr += 2
        if T == 0:
            type0.append(X)
        elif T == 1:
            type1.append(X)
        else:
            type2.append(X)
    
    type0.sort(reverse=True)
    type1.sort(reverse=True)
    type2.sort(reverse=True)
    
    sum0 = [0]
    for x in type0:
        sum0.append(sum0[-1] + x)
    
    sum1 = [0]
    for x in type1:
        sum1.append(sum1[-1] + x)
    
    sum2 = [0]
    for x in type2:
        sum2.append(sum2[-1] + x)
    
    t0 = len(type0)
    t1 = len(type1)
    t2 = len(type2)
    
    max_total = 0
    K_low = max(0, M - t2)
    K_high = min(M, t0 + t1)
    
    for K in range(K_low, K_high + 1):
        c = M - K
        if c < 0 or c > t2:
            continue
        S = sum2[c] if c <= t2 else sum2[-1]
        a_min = max(0, K - t1)
        a_max = min(K, t0)
        if a_min > a_max:
            continue
        
        left = a_min
        right = a_max
        best = 0
        
        while right - left > 5:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            
            b1 = K - mid1
            use1 = min(b1, S)
            use1 = min(use1, t1)
            s1 = sum0[mid1] + (sum1[use1] if use1 <= t1 else sum1[t1])
            
            b2 = K - mid2
            use2 = min(b2, S)
            use2 = min(use2, t1)
            s2 = sum0[mid2] + (sum1[use2] if use2 <= t1 else sum1[t1])
            
            if s1 < s2:
                left = mid1 + 1
            else:
                right = mid2 - 1
            
            if s1 > best:
                best = s1
            if s2 > best:
                best = s2
        
        for a in range(left, right + 1):
            if a < a_min or a > a_max:
                continue
            b = K - a
            if b < 0 or b > t1:
                continue
            use = min(b, S)
            use = min(use, t1)
            current = sum0[a] + sum1[use]
            if current > best:
                best = current
        
        if best > max_total:
            max_total = best
    
    print(max_total)

if __name__ == '__main__':
    main()