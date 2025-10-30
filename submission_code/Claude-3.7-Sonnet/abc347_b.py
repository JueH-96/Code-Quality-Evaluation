# YOUR CODE HERE
s = input().strip()
unique_substrings = set()
n = len(s)

for i in range(n):
    for j in range(i + 1, n + 1):
        unique_substrings.add(s[i:j])

print(len(unique_substrings))