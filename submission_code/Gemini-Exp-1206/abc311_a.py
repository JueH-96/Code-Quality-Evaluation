def solve():
    n = int(input())
    s = input()
    
    a_count = 0
    b_count = 0
    c_count = 0
    
    for i in range(n):
        if s[i] == 'A':
            a_count += 1
        elif s[i] == 'B':
            b_count += 1
        else:
            c_count += 1
        
        if a_count > 0 and b_count > 0 and c_count > 0:
            print(i + 1)
            return

solve()