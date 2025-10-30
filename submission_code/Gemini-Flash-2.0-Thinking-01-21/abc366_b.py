def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    m = 0
    for si in s:
        m = max(m, len(si))
    t = [""] * m
    for j in range(m):
        t_list = []
        for i in range(n):
            s_index = n - 1 - i
            if j < len(s[s_index]):
                t_list.append(s[s_index][j])
            else:
                t_list.append('*')
        t_str = "".join(t_list)
        while t_str.endswith("*") and len(t_str) > 1:
            t_str = t_str[:-1]
        t[j] = t_str
    for ti in t:
        print(ti)

if __name__ == '__main__':
    solve()