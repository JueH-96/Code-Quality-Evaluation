# YOUR CODE HERE
S = input().strip()
A = []
count = 0
for char in S:
    if char == '-':
        count += 1
    elif char == '|':
        if count != 0:
            A.append(count)
            count = 0
print(' '.join(map(str, A)))