def solve():
    n, m, k = map(int, input().split())
    x = list(map(int, input().split()))
    
    def is_subsequence(sub, seq):
        i = 0
        j = 0
        while i < len(sub) and j < len(seq):
            if sub[i] == seq[j]:
                i += 1
            j += 1
        return i == len(sub)
    
    count = 0
    mod = 998244353
    
    for i in range(k**n):
        seq = []
        temp = i
        for _ in range(n):
            seq.append(temp % k + 1)
            temp //= k
        seq = seq[::-1]
        
        possible = True
        for j in range(k**m):
            sub = []
            temp = j
            for _ in range(m):
                sub.append(temp % k + 1)
                temp //= k
            sub = sub[::-1]
            
            if not is_subsequence(sub, seq) and sub == x:
                possible = False
                break
        
        if possible:
            count = (count + 1) % mod
            
    print(count)

solve()