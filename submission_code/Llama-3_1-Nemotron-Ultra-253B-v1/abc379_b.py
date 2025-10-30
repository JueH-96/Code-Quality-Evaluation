n, k = map(int, input().split())
s = list(input().strip())
count = 0
i = 0
while i <= len(s) - k:
    if all(s[j] == 'O' for j in range(i, i + k)):
        count += 1
        for j in range(i, i + k):
            s[j] = 'X'
        i += k
    else:
        i += 1
print(count)