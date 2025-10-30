S = input().strip()
T = input().strip()

result = []
i = 0
n = len(S)
m = len(T)

for j in range(m):
    if i < n and T[j] == S[i]:
        result.append(j + 1)  # Convert to 1-based index
        i += 1

print(' '.join(map(str, result)))