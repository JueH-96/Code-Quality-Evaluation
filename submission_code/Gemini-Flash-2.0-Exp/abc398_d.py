def solve():
    n, r, c = map(int, input().split())
    s = input()
    
    smoke = set()
    smoke.add((0, 0))
    
    result = ""
    
    for i in range(n):
        new_smoke = set()
        for x, y in smoke:
            if s[i] == 'N':
                new_smoke.add((x - 1, y))
            elif s[i] == 'W':
                new_smoke.add((x, y - 1))
            elif s[i] == 'S':
                new_smoke.add((x + 1, y))
            elif s[i] == 'E':
                new_smoke.add((x, y + 1))
        
        smoke = new_smoke
        
        if (0, 0) not in smoke:
            smoke.add((0, 0))
        
        if (r, c) in smoke:
            result += "1"
        else:
            result += "0"
            
    print(result)

solve()