# YOUR CODE HERE
def solve():
    l, r = map(int, input().split())
    
    ans = []
    cur_l = l
    
    while cur_l < r:
        best_r = -1
        best_i = -1
        
        for i in range(61):
            j = cur_l // (2**i)
            if cur_l % (2**i) != 0:
                continue
                
            potential_r = (j + 1) * (2**i)
            
            if potential_r <= r and (best_r == -1 or potential_r < best_r):
                best_r = potential_r
                best_i = i
        
        if best_r == -1:
            for i in range(61):
                j = cur_l // (2**i)
                potential_r = (j + 1) * (2**i)
                if potential_r <= r and (best_r == -1 or potential_r < best_r):
                    best_r = potential_r
                    best_i = i
        
        ans.append((cur_l, best_r))
        cur_l = best_r
        
    print(len(ans))
    for l_val, r_val in ans:
        print(l_val, r_val)

solve()