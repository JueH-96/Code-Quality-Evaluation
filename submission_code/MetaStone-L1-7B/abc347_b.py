S = input().strip()
substrings = set()

n = len(S)
for i in range(n):
    for j in range(i, n):
        substrings.add(S[i:j+1])

print(len(substrings))