def find_middle_brother():
    # Read input from stdin
    s_ab, s_ac, s_bc = input().split()

    # Determine the order of the brothers
    if s_ab == '<' and s_ac == '<' and s_bc == '<':
        return 'B'
    elif s_ab == '<' and s_ac == '<' and s_bc == '>':
        return 'C'
    elif s_ab == '<' and s_ac == '>' and s_bc == '<':
        return 'A'
    elif s_ab == '<' and s_ac == '>' and s_bc == '>':
        return 'B'
    elif s_ab == '>' and s_ac == '<' and s_bc == '<':
        return 'A'
    elif s_ab == '>' and s_ac == '<' and s_bc == '>':
        return 'C'
    elif s_ab == '>' and s_ac == '>' and s_bc == '<':
        return 'B'
    elif s_ab == '>' and s_ac == '>' and s_bc == '>':
        return 'C'

# Print the result to stdout
print(find_middle_brother())