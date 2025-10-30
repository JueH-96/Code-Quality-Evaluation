# YOUR CODE HERE
def solve():
    s = input()
    x = input()
    y = input()

    n = len(x)
    m = len(y)

    def f(s, t, x):
        res = ""
        for char in x:
            if char == '0':
                res += s
            else:
                res += t
        return res

    for lt in range(0, len(s) + 1):
        
        
        lx0 = x.count('0')
        ly0 = y.count('0')

        lx1 = n - lx0
        ly1 = m - ly0

        if lx0 * len(s) > ly0 * len(s) + ly1 * lt:
            continue
        if ly0 * len(s) > lx0 * len(s) + lx1 * lt:
            continue
        
        
        if lx1 == 0 and ly1 == 0:
            if lx0 * len(s) == ly0 * len(s):
                print("Yes")
                return
            else:
                continue
        
        if lx1 == 0:
            if (ly0 * len(s) - lx0 * len(s)) % ly1 == 0:
                lt_calc = (ly0 * len(s) - lx0 * len(s)) // ly1
                if 0 <= lt_calc <= len(s):
                    
                    t = ""
                    fx = f(s, t, x)
                    fy = f(s, t, y)
                    
                    if len(fx) == len(fy):
                        print("Yes")
                        return
        elif ly1 == 0:
            if (lx0 * len(s) - ly0 * len(s)) % lx1 == 0:
                lt_calc = (lx0 * len(s) - ly0 * len(s)) // lx1
                if 0 <= lt_calc <= len(s):
                    t = ""
                    fx = f(s, t, x)
                    fy = f(s, t, y)
                    if len(fx) == len(fy):
                        print("Yes")
                        return
        else:
            if (ly0 * len(s) - lx0 * len(s)) % (ly1 - lx1) == 0:
                lt_calc = (ly0 * len(s) - lx0 * len(s)) // (ly1 - lx1)
                if 0 <= lt_calc <= len(s):
                    t = ""
                    fx = f(s, t, x)
                    fy = f(s, t, y)
                    if len(fx) == len(fy):
                        print("Yes")
                        return
    print("No")


t = int(input())
for _ in range(t):
    solve()