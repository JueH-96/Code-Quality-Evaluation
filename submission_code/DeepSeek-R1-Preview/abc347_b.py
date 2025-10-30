s = input().strip()
unique_substrings = set()
n = len(s)
for i in range(n):
    for j in range(i, n):
        substr = s[i:j+1]
        unique_substrings.add(substr)
print(len(unique_substrings))