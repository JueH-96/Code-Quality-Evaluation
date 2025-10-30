S = input().strip()
count = 0
i = 0
n = len(S)
while i < n:
    if S[i] != '0':
        count += 1
        i += 1
    else:
        zeros = 0
        while i < n and S[i] == '0':
            zeros += 1
            i += 1
        count += (zeros + 1) // 2
print(count)