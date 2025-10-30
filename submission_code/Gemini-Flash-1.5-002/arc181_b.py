def solve():
    s = input()
    x = input()
    y = input()

    n = len(x)
    m = len(y)

    if n == 0 and m == 0:
        print("Yes")
        return
    if n == 0 or m == 0:
        print("No")
        return

    def f(s, t, x):
        res = ""
        for char in x:
            if char == '0':
                res += s
            else:
                res += t
        return res

    
    def check(t):
        return f(s, t, x) == f(s, t, y)

    
    if check(""):
        print("Yes")
        return

    
    prefix_x = ""
    prefix_y = ""
    
    i = 0
    while i < n and i < m and x[i] == y[i]:
        if x[i] == '0':
            prefix_x += s
            prefix_y += s
        else:
            
            pass
        i+=1
    
    
    if i == n and i == m:
        print("Yes")
        return
    
    if i == n or i == m:
        print("No")
        return
    
    
    if x[i] == '0' and y[i] == '1':
        
        len_s = len(s)
        
        
        rem_x = x[i:]
        rem_y = y[i:]
        
        
        num_zeros_x = rem_x.count('0')
        num_ones_x = rem_x.count('1')
        
        num_zeros_y = rem_y.count('0')
        num_ones_y = rem_y.count('1')
        
        if num_zeros_x == num_zeros_y and num_ones_x == num_ones_y:
            print("Yes")
        else:
            print("No")
        
        return
    
    
    print("No")


T = int(input())
for _ in range(T):
    solve()