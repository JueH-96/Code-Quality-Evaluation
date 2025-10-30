# YOUR CODE HERE
def solve():
    s = input()
    t = input()
    n = len(s)
    s_counts = {}
    t_counts = {}
    atcoder = "atcoder"

    for i in range(n):
        if s[i] != '@':
            s_counts[s[i]] = s_counts.get(s[i], 0) + 1
        if t[i] != '@':
            t_counts[t[i]] = t_counts.get(t[i], 0) + 1
    
    for char in "atcoder":
        if s_counts.get(char, 0) > t_counts.get(char, 0):
            print("No")
            return
    
    s_at = s.count('@')
    t_at = t.count('@')

    
    diff = 0
    for char in "atcoder":
      diff += t_counts.get(char, 0) - s_counts.get(char,0)

    if s_at >= diff:
        print("Yes")
    else:
        print("No")

solve()