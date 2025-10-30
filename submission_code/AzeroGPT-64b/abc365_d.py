from collections import defaultdict
n = int(input())
s = input()
B = defaultdict(int)
B[s[0]] += 1
for i in range(1, n):
    B[s[i]] += 1
    if s[i] == s[i - 1]:
        B[s[i]] -= 1
print(sum([max(0, B[k] - 1) for k in 'RPS']))