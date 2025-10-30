# Read the input
S_AB, S_AC, S_BC = input().split()

# Determine the middle brother
if S_AB == '<':
    if S_BC == '<':
        middle_brother = 'B'
    else:
        middle_brother = 'A'
else:
    if S_AC == '>':
        middle_brother = 'A'
    else:
        middle_brother = 'C'

# Print the middle brother
print(middle_brother)