def solve():
    L, R = map(int, input().split())
    
    result = []
    
    while L < R:
        i = 0
        while True:
            power_of_2 = 2**i
            if L % power_of_2 == 0 and L + power_of_2 <= R and (L // power_of_2) % 2 == 0:
                i += 1
            else:
                i -= 1
                break
        
        if i == -1:
            i = 0
            power_of_2 = 1
            while L % power_of_2 != 0:
                power_of_2 *= 2
                i += 1
            i -= 1
            power_of_2 = 2**i
            if i == -1:
                power_of_2 = 1
        else:
            power_of_2 = 2**i
        
        result.append((L, L + power_of_2))
        L += power_of_2
    
    print(len(result))
    for l, r in result:
        print(l, r)

solve()