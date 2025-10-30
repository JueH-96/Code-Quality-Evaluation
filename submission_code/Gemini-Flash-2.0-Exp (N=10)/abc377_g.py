def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    for k in range(n):
        t = s[k]
        min_cost = len(t)
        
        for i in range(k):
            other = s[i]
            
            lcp = 0
            for j in range(min(len(t), len(other))):
                if t[j] == other[j]:
                    lcp += 1
                else:
                    break
            
            cost = len(t) - lcp + len(other) - lcp
            min_cost = min(min_cost, cost)
            
        print(min_cost)

solve()