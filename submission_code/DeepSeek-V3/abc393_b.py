S = input().strip()
count = 0
n = len(S)
for i in range(n):
    for d in range(1, n):
        j = i + d
        k = j + d
        if k >= n:
            break
        if S[i] == 'A' and S[j] == 'B' and S[k] == 'C':
            count += 1
print(count)