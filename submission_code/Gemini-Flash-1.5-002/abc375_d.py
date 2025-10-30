S = input()
n = len(S)
count = 0
for i in range(n - 2):
    for j in range(i + 1, n - 1):
        for k in range(j + 1, n):
            if S[i] == S[k] and i < j < k:
                count += 1
print(count)