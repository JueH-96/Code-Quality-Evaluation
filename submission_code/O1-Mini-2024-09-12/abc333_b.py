def solve():
    import sys
    mapping = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    
    lines = sys.stdin.read().split()
    s = lines[0]
    t = lines[1]
    
    def step(a, b):
        d = abs(a - b)
        return min(d, 5 - d)
    
    s1, s2 = mapping[s[0]], mapping[s[1]]
    t1, t2 = mapping[t[0]], mapping[t[1]]
    
    s_step = step(s1, s2)
    t_step = step(t1, t2)
    
    if (s_step in [1,4] and t_step in [1,4]) or (s_step in [2,3] and t_step in [2,3]):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()