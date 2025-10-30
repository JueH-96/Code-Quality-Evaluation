# Read the input from stdin
s_ab, s_ac, s_bc = input().split()

# Determine the order of the brothers
if s_ab == '<':
    if s_ac == '<':
        if s_bc == '<':
            print('B')
        else:
            print('C')
    else:
        print('C')
else:
    if s_ac == '<':
        print('B')
    else:
        print('A')