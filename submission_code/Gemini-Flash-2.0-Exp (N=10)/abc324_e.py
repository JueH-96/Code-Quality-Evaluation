def is_subsequence(s, t):
    i = 0
    j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            j += 1
        i += 1
    return j == len(t)

def solve():
    n, t = input().split()
    n = int(n)
    s_list = [input() for _ in range(n)]
    
    count = 0
    for i in range(n):
        for j in range(n):
            if is_subsequence(s_list[i] + s_list[j], t):
                count += 1
    print(count)

solve()