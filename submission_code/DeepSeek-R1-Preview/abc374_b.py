s = input().strip()
t = input().strip()

max_length = max(len(s), len(t))

for i in range(1, max_length + 1):
    if i > len(s) or i > len(t):
        print(i)
        exit()
    if s[i-1] != t[i-1]:
        print(i)
        exit()

print(0)