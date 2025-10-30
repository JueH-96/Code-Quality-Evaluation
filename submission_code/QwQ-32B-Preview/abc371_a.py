s_ab, s_ac, s_bc = input().split()

if s_ab == '<' and s_ac == '<':
    if s_bc == '<':
        middle = 'B'
    else:
        middle = 'C'
elif s_ab == '<' and s_ac == '>':
    middle = 'A'
elif s_ab == '>' and s_ac == '<':
    middle = 'A'
else:  # s_ab == '>' and s_ac == '>'
    if s_bc == '<':
        middle = 'C'
    else:
        middle = 'B'

print(middle)