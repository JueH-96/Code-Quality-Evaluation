def solve():
    s = input()
    x = input()
    y = input()
    
    if not s:
        print("Yes")
        return
        
    if x == y:
        print("Yes")
        return
        
    len_s = len(s)
    count_0_x = x.count('0')
    count_1_x = len(x) - count_0_x
    count_0_y = y.count('0')
    count_1_y = len(y) - count_0_y
    
    delta_0 = count_0_x - count_0_y
    delta_1 = count_1_x - count_1_y
    
    if delta_1 == 0:
        if delta_0 == 0:
            print("Yes")
        else:
            print("No")
        return
        
    t_len_val = - (delta_0 * len_s) / delta_1
    
    if t_len_val < 0 or t_len_val != int(t_len_val):
        print("No")
        return
        
    t_len = int(t_len_val)
    
    if t_len == 0:
        t = ""
    else:
        t = "a" * t_len
        
    fx = ""
    for char_x in x:
        if char_x == '0':
            fx += s
        else:
            fx += t
            
    fy = ""
    for char_y in y:
        if char_y == '0':
            fy += s
        else:
            fy += t
            
    if fx == fy:
        print("Yes")
    else:
        print("No")

t = int(input())
for _ in range(t):
    solve()