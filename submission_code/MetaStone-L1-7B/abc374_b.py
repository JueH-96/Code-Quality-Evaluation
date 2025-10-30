s = input().strip()
t = input().strip()

m = len(s)
n = len(t)

max_len = max(m, n)

for i in range(1, max_len + 1):
    if i > m or i > n:
        print(i)
        exit()
    else:
        if s[i-1] != t[i-1]:
            print(i)
            exit()

print(0)