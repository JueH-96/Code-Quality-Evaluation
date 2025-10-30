# YOUR CODE HERE
S = input()
n = len(S)
distinct_substrings = set()

for i in range(n):
    for j in range(i, n):
        # Extract substring S[i...j]
        # In Python, S[from_index : to_index_plus_1]
        substring = S[i : j+1]
        distinct_substrings.add(substring)

print(len(distinct_substrings))