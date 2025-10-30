# Read input
s_ab, s_ac, s_bc = input().split()

# Count how many people each brother is older than
older_count = {'A': 0, 'B': 0, 'C': 0}

# Process S_AB
if s_ab == '<':  # A is younger than B
    older_count['B'] += 1
else:  # A is older than B
    older_count['A'] += 1

# Process S_AC
if s_ac == '<':  # A is younger than C
    older_count['C'] += 1
else:  # A is older than C
    older_count['A'] += 1

# Process S_BC
if s_bc == '<':  # B is younger than C
    older_count['C'] += 1
else:  # B is older than C
    older_count['B'] += 1

# Find who is older than exactly 1 person (the middle brother)
for brother, count in older_count.items():
    if count == 1:
        print(brother)
        break