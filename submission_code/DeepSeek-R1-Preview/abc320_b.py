s = input().strip()
n = len(s)
max_length = 1

for l in range(n, 1, -1):
    for i in range(n - l + 1):
        substr = s[i:i+l]
        if substr == substr[::-1]:
            print(l)
            exit()
print(max_length)