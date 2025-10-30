# YOUR CODE HERE
S = input()
substr_set = set()
n = len(S)
for i in range(n):
    for j in range(i + 1, n + 1):
        substr_set.add(S[i:j])
print(len(substr_set))