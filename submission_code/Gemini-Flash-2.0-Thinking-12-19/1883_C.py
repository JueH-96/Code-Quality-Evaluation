def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    if k == 2:
        has_even = False
        for x in a:
            if x % 2 == 0:
                has_even = True
                break
        if has_even:
            print(0)
        else:
            print(1)
    elif k == 3:
        has_div_by_3 = False
        has_rem_2 = False
        for x in a:
            if x % 3 == 0:
                has_div_by_3 = True
                break
            if x % 3 == 2:
                has_rem_2 = True
                
        if has_div_by_3:
            print(0)
        elif has_rem_2:
            print(1)
        else:
            print(2)
    elif k == 4:
        v2_sum = 0
        for x in a:
            v = 0
            while x > 0 and x % 2 == 0:
                v += 1
                x //= 2
            v2_sum += v
            
        if v2_sum >= 2:
            print(0)
        elif v2_sum == 1:
            print(1)
        else: # v2_sum == 0
            has_rem_3_mod_4 = False
            for x in a:
                if x % 2 != 0 and x % 4 == 3:
                    has_rem_3_mod_4 = True
                    break
            if has_rem_3_mod_4:
                print(1)
            else:
                print(2)
    elif k == 5:
        has_div_by_5 = False
        has_rem_4 = False
        has_rem_3 = False
        has_rem_2 = False
        for x in a:
            if x % 5 == 0:
                has_div_by_5 = True
                break
            if x % 5 == 4:
                has_rem_4 = True
            if x % 5 == 3:
                has_rem_3 = True
            if x % 5 == 2:
                has_rem_2 = True
                
        if has_div_by_5:
            print(0)
        elif has_rem_4:
            print(1)
        elif has_rem_3:
            print(2)
        elif has_rem_2:
            print(3)
        else:
            print(4)

t = int(input())
for _ in range(t):
    solve()