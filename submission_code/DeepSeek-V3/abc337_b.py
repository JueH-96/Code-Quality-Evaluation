# YOUR CODE HERE
S = input().strip()
n = len(S)
if n == 0:
    print("Yes")
    exit()
# Find the transition points
a_end = -1
b_start = -1
b_end = -1
c_start = -1
# Find the end of A's
for i in range(n):
    if S[i] != 'A':
        a_end = i-1
        break
else:
    a_end = n-1
# Find the start of B's
for i in range(a_end+1, n):
    if S[i] == 'B':
        b_start = i
        break
else:
    b_start = n
# Find the end of B's
for i in range(b_start, n):
    if S[i] != 'B':
        b_end = i-1
        break
else:
    b_end = n-1
# Find the start of C's
for i in range(b_end+1, n):
    if S[i] == 'C':
        c_start = i
        break
else:
    c_start = n
# Check if the remaining characters are all C's
for i in range(c_start, n):
    if S[i] != 'C':
        print("No")
        exit()
# Check if the order is correct
if a_end < b_start and b_end < c_start:
    print("Yes")
else:
    print("No")