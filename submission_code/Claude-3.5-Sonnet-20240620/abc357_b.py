# YOUR CODE HERE
S = input().strip()

uppercase_count = sum(1 for char in S if char.isupper())
lowercase_count = len(S) - uppercase_count

if uppercase_count > lowercase_count:
    print(S.upper())
else:
    print(S.lower())