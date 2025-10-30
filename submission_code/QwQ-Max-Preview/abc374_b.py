s = input().strip()
t = input().strip()

max_len = max(len(s), len(t))

for i in range(1, max_len + 1):
    if i > len(s):
        print(i)
        break
    elif i > len(t):
        print(i)
        break
    else:
        if s[i-1] != t[i-1]:
            print(i)
            break
else:
    print(0)