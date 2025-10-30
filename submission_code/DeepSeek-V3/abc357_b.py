# YOUR CODE HERE
S = input().strip()
lower_count = 0
upper_count = 0

for char in S:
    if char.islower():
        lower_count += 1
    else:
        upper_count += 1

if upper_count > lower_count:
    print(S.upper())
else:
    print(S.lower())