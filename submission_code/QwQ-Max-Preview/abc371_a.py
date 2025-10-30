# Read the input relations
s_ab, s_ac, s_bc = input().split()

# Calculate the number of older brothers for each
count_a = (s_ab == '<') + (s_ac == '<')
count_b = (s_ab == '>') + (s_bc == '<')
count_c = (s_ac == '>') + (s_bc == '>')

# Determine the middle brother (count of 1)
if count_a == 1:
    print('A')
elif count_b == 1:
    print('B')
else:
    print('C')