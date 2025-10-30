# YOUR CODE HERE
s_ab, s_ac, s_bc = input().split()

# Count how many others each brother is older than
count_a = 0
count_b = 0
count_c = 0

# Check A vs B
if s_ab == '>':  # A is older than B
    count_a += 1
else:  # A is younger than B
    count_b += 1

# Check A vs C
if s_ac == '>':  # A is older than C
    count_a += 1
else:  # A is younger than C
    count_c += 1

# Check B vs C
if s_bc == '>':  # B is older than C
    count_b += 1
else:  # B is younger than C
    count_c += 1

# The middle brother is the one who is older than exactly 1 other
if count_a == 1:
    print('A')
elif count_b == 1:
    print('B')
else:
    print('C')