s = input().strip()
substrings = set()
n = len(s)
for i in range(n):
    for j in range(i, n):
        substrings.add(s[i:j+1])
print(len(substrings))