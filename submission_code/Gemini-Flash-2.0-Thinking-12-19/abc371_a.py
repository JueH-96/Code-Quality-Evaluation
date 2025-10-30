s_ab, s_ac, s_bc = input().split()

older_than_count = {'A': 0, 'B': 0, 'C': 0}

if s_ab == '>':
    older_than_count['A'] += 1
else:
    older_than_count['B'] += 1

if s_ac == '>':
    older_than_count['A'] += 1
else:
    older_than_count['C'] += 1

if s_bc == '>':
    older_than_count['B'] += 1
else:
    older_than_count['C'] += 1

if older_than_count['A'] == 1:
    print("A")
elif older_than_count['B'] == 1:
    print("B")
else:
    print("C")