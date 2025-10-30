# YOUR CODE HERE
S = input().strip()
substrings = set()

n = len(S)
for i in range(n):
    for j in range(i, n):
        substring = S[i:j+1]
        substrings.add(substring)

print(len(substrings))