def solve():
    n = int(input())
    k = list(map(int, input().split()))
    
    min_max_group_size = float('inf')
    
    for i in range(2**n):
        group_a = []
        group_b = []
        
        for j in range(n):
            if (i >> j) & 1:
                group_a.append(k[j])
            else:
                group_b.append(k[j])
        
        sum_a = sum(group_a)
        sum_b = sum(group_b)
        
        max_group_size = max(sum_a, sum_b)
        min_max_group_size = min(min_max_group_size, max_group_size)
    
    print(min_max_group_size)

solve()