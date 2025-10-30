s = input().strip()
sab, sac, sbc = s.split()

# Create a dictionary to store relative positions
positions = {'A': 0, 'B': 0, 'C': 0}

# Count how many times each person is older than others
if sab == '>':
    positions['A'] += 1
else:
    positions['B'] += 1
    
if sac == '>':
    positions['A'] += 1
else:
    positions['C'] += 1
    
if sbc == '>':
    positions['B'] += 1
else:
    positions['C'] += 1

# The one with exactly 1 person being older than them is the middle
for person, count in positions.items():
    if count == 1:
        print(person)
        break