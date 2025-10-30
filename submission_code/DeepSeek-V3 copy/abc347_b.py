S = input().strip()
unique_substrings = set()
n = len(S)
for i in range(n):
    for j in range(i+1, n+1):
        substring = S[i:j]
        unique_substrings.add(substring)
print(len(unique_substrings))