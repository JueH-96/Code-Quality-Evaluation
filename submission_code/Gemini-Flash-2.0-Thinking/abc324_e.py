def is_subsequence(text, pattern):
    i = 0
    j = 0
    while i < len(text) and j < len(pattern):
        if text[i] == pattern[j]:
            j += 1
        i += 1
    return j == len(pattern)

def solve():
    n_str, t = input().split()
    n = int(n_str)
    s_list = [input() for _ in range(n)]

    count = 0
    for i in range(n):
        for j in range(n):
            concatenated = s_list[i] + s_list[j]
            if is_subsequence(concatenated, t):
                count += 1
    print(count)

solve()