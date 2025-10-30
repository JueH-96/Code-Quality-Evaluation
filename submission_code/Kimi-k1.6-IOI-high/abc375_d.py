from collections import defaultdict

S = input().strip()
char_info = defaultdict(lambda: [0, 0])  # [count, sum of indices]
total = 0

for idx in range(len(S)):
    c = S[idx]
    cnt, summ = char_info[c]
    contribution = cnt * (idx - 1) - summ
    total += contribution
    char_info[c][0] += 1
    char_info[c][1] += idx

print(total)