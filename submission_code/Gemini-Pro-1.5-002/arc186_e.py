def solve():
    n, m, k = map(int, input().split())
    x = list(map(int, input().split()))
    
    mod = 998244353
    
    def count_subsequences(arr):
        subsequences = set()
        for i in range(1 << len(arr)):
            sub = []
            for j in range(len(arr)):
                if (i >> j) & 1:
                    sub.append(arr[j])
            subsequences.add(tuple(sub))
        return subsequences

    def is_subsequence(a, b):
        i = 0
        j = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
            j += 1
        return i == len(a)

    ans = 0
    for i in range(k**n):
        a = []
        temp = i
        for _ in range(n):
            a.append(temp % k + 1)
            temp //= k
        
        
        forbidden_found = False
        for j in range(k**m):
            b = []
            temp_b = j
            for _ in range(m):
                b.append(temp_b % k + 1)
                temp_b //= k
            
            if not is_subsequence(b, a):
                if b == x:
                    forbidden_found = True
                else:
                    forbidden_found = False
                    break
        
        if forbidden_found:
            ans = (ans + 1) % mod

    print(ans)

solve()