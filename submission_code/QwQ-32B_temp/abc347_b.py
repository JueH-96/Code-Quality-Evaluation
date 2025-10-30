s = input().strip()
substrings = set()

for i in range(len(s)):
    for j in range(i, len(s)):
        substr = s[i:j+1]
        substrings.add(substr)

print(len(substrings))