S = input().strip()
unique_substrings = set()

for i in range(len(S)):
    for j in range(i+1, len(S)+1):
        substring = S[i:j]
        unique_substrings.add(substring)

print(len(unique_substrings))