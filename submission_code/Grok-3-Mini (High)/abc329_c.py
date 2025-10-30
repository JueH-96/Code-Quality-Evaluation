import sys
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]
len_s = len(S)
max_run = {}
i = 0
while i < len_s:
    curr_char = S[i]
    curr_len = 1
    i += 1
    while i < len_s and S[i] == curr_char:
        curr_len += 1
        i += 1
    max_run[curr_char] = max(max_run.get(curr_char, 0), curr_len)
total = sum(max_run.values())
print(total)