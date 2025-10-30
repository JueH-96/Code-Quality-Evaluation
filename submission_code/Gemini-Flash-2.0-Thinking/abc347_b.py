s = input()
n = len(s)
substrings = set()
for i in range(n):
    for j in range(i, n):
        substring = s[i:j+1]
        substrings.add(substring)
print(len(substrings))