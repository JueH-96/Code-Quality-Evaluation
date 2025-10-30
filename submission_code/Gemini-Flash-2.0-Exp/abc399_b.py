def solve():
    n = int(input())
    p = list(map(int, input().split()))
    
    ranks = [0] * n
    r = 1
    undetermined = list(range(n))
    
    while undetermined:
        max_score = -1
        for i in undetermined:
            max_score = max(max_score, p[i])
        
        k = 0
        to_rank = []
        for i in undetermined:
            if p[i] == max_score:
                k += 1
                to_rank.append(i)
                
        for i in to_rank:
            ranks[i] = r
            
        r += k
        
        new_undetermined = []
        for i in undetermined:
            if i not in to_rank:
                new_undetermined.append(i)
        undetermined = new_undetermined
        
    for rank in ranks:
        print(rank)

solve()