S = input().strip()
unique_substrings = set()
n = len(S)
for i in range(n):
    for j in range(i+1, n+1):
        unique_substrings.add(S[i:j])
print(len(unique_substrings))