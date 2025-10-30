s = input().strip()
t = input().strip()

min_len = min(len(s), len(t))
for i in range(min_len):
    if s[i] != t[i]:
        print(i + 1)
        exit()
if len(s) != len(t):
    print(min_len + 1)
else:
    print(0)