def solve():
    l, r = map(int, input().split())
    
    result = []
    while l < r:
        best_len = 1
        best_next_l = l + 1
        
        for i in range(61):
            power_of_2 = 1 << i
            
            if l % power_of_2 == 0:
                next_l = l + power_of_2
                if next_l <= r:
                    if next_l - l > best_len:
                        best_len = next_l - l
                        best_next_l = next_l
        
        result.append((l, best_next_l))
        l = best_next_l
    
    print(len(result))
    for start, end in result:
        print(start, end)

solve()