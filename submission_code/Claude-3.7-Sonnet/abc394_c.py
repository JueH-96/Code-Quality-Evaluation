# YOUR CODE HERE
S = input().strip()

# Keep replacing the leftmost "WA" with "AC" until no "WA" remains
while "WA" in S:
    S = S.replace("WA", "AC", 1)  # The 1 ensures only the leftmost occurrence is replaced

print(S)