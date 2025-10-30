def is_subsequence(s, t):
    t_ptr = 0
    s_ptr = 0
    while s_ptr < len(s) and t_ptr < len(t):
        if s[s_ptr] == t[t_ptr]:
            t_ptr += 1
        s_ptr += 1
    return t_ptr == len(t)

def solve():
    n, t = input().split()
    n = int(n)
    s_list = [input() for _ in range(n)]
    count = 0
    for i in range(n):
        for j in range(n):
            concatenated_string = s_list[i] + s_list[j]
            if is_subsequence(concatenated_string, t):
                count += 1
    print(count)

if __name__ == '__main__':
    solve()