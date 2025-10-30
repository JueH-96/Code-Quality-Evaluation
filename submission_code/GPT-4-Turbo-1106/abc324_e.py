def is_subsequence(s, t):
    t_idx = 0
    for char in s:
        if t_idx < len(t) and char == t[t_idx]:
            t_idx += 1
    return t_idx == len(t)

def count_pairs(strings, target):
    count = 0
    n = len(strings)
    for i in range(n):
        for j in range(n):
            if is_subsequence(strings[i] + strings[j], target):
                count += 1
    return count

# Read input
N, T = input().split()
N = int(N)
strings = [input().strip() for _ in range(N)]

# Solve and output
print(count_pairs(strings, T))