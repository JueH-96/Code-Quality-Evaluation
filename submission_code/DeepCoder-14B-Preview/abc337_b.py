s = input().strip()

# Find the end of A's
a_end = 0
while a_end < len(s) and s[a_end] == 'A':
    a_end += 1

# Find the end of B's starting from a_end
b_start = a_end
b_end = b_start
while b_end < len(s) and s[b_end] == 'B':
    b_end += 1

# Find the end of C's starting from b_end
c_start = b_end
c_end = c_start
while c_end < len(s) and s[c_end] == 'C':
    c_end += 1

if c_end == len(s):
    print("Yes")
else:
    print("No")