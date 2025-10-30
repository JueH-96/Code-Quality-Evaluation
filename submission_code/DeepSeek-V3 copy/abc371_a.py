# Read the input
S_AB, S_AC, S_BC = input().split()

# Determine the age relationships
# We will represent the ages as a dictionary where the key is the brother and the value is their age rank
# We will assign ranks such that the oldest has the highest rank (3), middle has 2, youngest has 1

# Initialize the ranks
ranks = {'A': 0, 'B': 0, 'C': 0}

# Update ranks based on S_AB
if S_AB == '<':
    ranks['A'] += 1
else:
    ranks['B'] += 1

# Update ranks based on S_AC
if S_AC == '<':
    ranks['A'] += 1
else:
    ranks['C'] += 1

# Update ranks based on S_BC
if S_BC == '<':
    ranks['B'] += 1
else:
    ranks['C'] += 1

# Now, determine the middle brother
# The brother with rank 2 is the middle one
for brother, rank in ranks.items():
    if rank == 2:
        print(brother)
        break