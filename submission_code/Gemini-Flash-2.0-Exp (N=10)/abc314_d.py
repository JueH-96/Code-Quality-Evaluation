def solve():
    n = int(input())
    s = list(input())
    q = int(input())
    
    for _ in range(q):
        line = input().split()
        t = int(line[0])
        
        if t == 1:
            x = int(line[1]) - 1
            c = line[2]
            s[x] = c
        elif t == 2:
            s = [char.lower() for char in s]
        elif t == 3:
            s = [char.upper() for char in s]
            
    print("".join(s))

solve()