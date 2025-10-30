# Read the input from standard input
s_ab, s_ac, s_bc = input().split()

# Calculate the number of brothers each is older than
count_A = (1 if s_ab == '>' else 0) + (1 if s_ac == '>' else 0)
count_B = (1 if s_ab == '<' else 0) + (1 if s_bc == '>' else 0)
count_C = (1 if s_ac == '<' else 0) + (1 if s_bc == '<' else 0)

# Determine who has exactly one brother they are older than (middle brother)
if count_A == 1:
    print('A')
elif count_B == 1:
    print('B')
elif count_C == 1:
    print('C')