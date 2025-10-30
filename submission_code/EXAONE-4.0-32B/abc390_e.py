import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    X_val = int(next(it))
    groups = {1: [], 2: [], 3: []}
    for _ in range(n):
        v = int(next(it))
        a = int(next(it))
        c = int(next(it))
        groups[v].append((a, c))
    
    dp1 = [0] * (X_val + 1)
    for a, c in groups[1]:
        for j in range(X_val, c - 1, -1):
            if dp1[j] < dp1[j - c] + a:
                dp1[j] = dp1[j - c] + a
    
    dp2 = [0] * (X_val + 1)
    for a, c in groups[2]:
        for j in range(X_val, c - 1, -1):
            if dp2[j] < dp2[j - c] + a:
                dp2[j] = dp2[j - c] + a
    
    dp3 = [0] * (X_val + 1)
    for a, c in groups[3]:
        for j in range(X_val, c - 1, -1):
            if dp3[j] < dp3[j - c] + a:
                dp3[j] = dp3[j - c] + a
    
    total1 = sum(a for a, _ in groups[1])
    total2 = sum(a for a, _ in groups[2])
    total3 = sum(a for a, _ in groups[3])
    T_high = min(total1, total2, total3)
    
    lo, hi = 0, T_high
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        c1 = float('inf')
        for j in range(X_val + 1):
            if dp1[j] >= mid:
                c1 = min(c1, j)
        c2 = float('inf')
        for j in range(X_val + 1):
            if dp2[j] >= mid:
                c2 = min(c2, j)
        c3 = float('inf')
        for j in range(X_val + 1):
            if dp3[j] >= mid:
                c3 = min(c3, j)
        
        if c1 + c2 + c3 <= X_val:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    
    print(ans)

if __name__ == "__main__":
    main()