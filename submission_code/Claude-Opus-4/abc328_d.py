# YOUR CODE HERE
S = input().strip()

while "ABC" in S:
    S = S.replace("ABC", "", 1)  # Replace only the first (leftmost) occurrence

print(S)