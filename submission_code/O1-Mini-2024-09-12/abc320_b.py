S = input().strip()
max_len = 1
n = len(S)

for i in range(n):
    for j in range(i + 1, n + 1):
        substring = S[i:j]
        if substring == substring[::-1]:
            if len(substring) > max_len:
                max_len = len(substring)

print(max_len)