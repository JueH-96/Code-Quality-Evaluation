S = input().strip()
count = 0
n = len(S)
for j in range(n):
    if S[j] != 'B':
        continue
    for i in range(j):
        if S[i] == 'A':
            k = 2 * j - i
            if k < n and S[k] == 'C':
                count += 1
print(count)