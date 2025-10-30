def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if len(input_data) < 3:
        return
    
    s_ab, s_ac, s_bc = input_data[0], input_data[1], input_data[2]
    
    # We'll count how many brothers each candidate is older than.
    older_count = {'A': 0, 'B': 0, 'C': 0}
    
    # For S_AB: if S_AB is '<', then A is younger than B, so B is older than A.
    # Otherwise, if S_AB is '>', then A is older than B.
    if s_ab == '<':
        older_count['B'] += 1
    else:   # s_ab == '>'
        older_count['A'] += 1
    
    # For S_AC: if S_AC is '<', then A is younger than C, so C is older than A.
    # Otherwise, if s_ac == '>', then A is older than C.
    if s_ac == '<':
        older_count['C'] += 1
    else:   # s_ac == '>'
        older_count['A'] += 1
    
    # For S_BC: if S_BC is '<', then B is younger than C, so C is older than B.
    # Otherwise, if s_bc == '>', then B is older than C.
    if s_bc == '<':
        older_count['C'] += 1
    else:   # s_bc == '>'
        older_count['B'] += 1
    
    # The middle brother is exactly the one who is older than 1 brother.
    for brother in older_count:
        if older_count[brother] == 1:
            sys.stdout.write(brother)
            return

if __name__ == '__main__':
    main()