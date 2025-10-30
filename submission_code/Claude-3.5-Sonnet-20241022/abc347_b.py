S = input().strip()
n = len(S)
substrings = set()

for i in range(n):
    for j in range(i+1, n+1):
        substrings.add(S[i:j])

print(len(substrings))