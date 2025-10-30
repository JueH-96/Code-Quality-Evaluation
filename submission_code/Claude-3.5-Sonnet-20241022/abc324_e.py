def is_subsequence(s, t):
    t_idx = 0
    t_len = len(t)
    
    for c in s:
        if t_idx < t_len and c == t[t_idx]:
            t_idx += 1
    
    return t_idx == t_len

N, T = input().split()
N = int(N)
strings = [input() for _ in range(N)]

count = 0
for i in range(N):
    for j in range(N):
        if is_subsequence(strings[i] + strings[j], T):
            count += 1

print(count)