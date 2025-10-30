# Read the input
s = input().strip().split()
sab, sac, sbc = s[0], s[1], s[2]

# Initialize counts for how many each brother is older than
counts = {'A': 0, 'B': 0, 'C': 0}

# Process each relationship
if sab == '<':
    counts['B'] += 1
else:
    counts['A'] += 1

if sac == '<':
    counts['C'] += 1
else:
    counts['A'] += 1

if sbc == '<':
    counts['C'] += 1
else:
    counts['B'] += 1

# Find the brother with exactly one person older than him
for brother in counts:
    if counts[brother] == 1:
        print(brother)
        break