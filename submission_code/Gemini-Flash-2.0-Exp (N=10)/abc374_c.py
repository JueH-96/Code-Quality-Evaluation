def solve():
    n = int(input())
    k = list(map(int, input().split()))
    
    min_max_group_size = float('inf')
    
    for i in range(2**n):
        group_a_size = 0
        group_b_size = 0
        
        for j in range(n):
            if (i >> j) & 1:
                group_a_size += k[j]
            else:
                group_b_size += k[j]
        
        max_group_size = max(group_a_size, group_b_size)
        min_max_group_size = min(min_max_group_size, max_group_size)
        
    print(min_max_group_size)

solve()