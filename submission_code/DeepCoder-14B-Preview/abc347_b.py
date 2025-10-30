s = input().strip()
substrings = set()
n = len(s)
for i in range(n):
    for l in range(1, n - i + 1):
        substr = s[i:i+l]
        substrings.add(substr)
print(len(substrings))