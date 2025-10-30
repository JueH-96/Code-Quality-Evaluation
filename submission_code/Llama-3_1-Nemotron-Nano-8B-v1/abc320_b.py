s = input().strip()
for l in range(len(s), 0, -1):
    for i in range(len(s) - l + 1):
        substr = s[i:i+l]
        if substr == substr[::-1]:
            print(l)
            exit()