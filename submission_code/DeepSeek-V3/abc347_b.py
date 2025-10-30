# YOUR CODE HERE
S = input().strip()
unique_substrings = set()
n = len(S)
for i in range(n):
    current_substring = ""
    for j in range(i, n):
        current_substring += S[j]
        unique_substrings.add(current_substring)
print(len(unique_substrings))