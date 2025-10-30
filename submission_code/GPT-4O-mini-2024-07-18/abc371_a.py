# YOUR CODE HERE
def find_middle_brother(S_AB, S_AC, S_BC):
    # Initialize the age order
    ages = {'A': 0, 'B': 0, 'C': 0}
    
    # Determine the ages based on the relationships
    if S_AB == '<':
        ages['A'] += 1  # A is younger than B
    else:
        ages['B'] += 1  # A is older than B

    if S_AC == '<':
        ages['A'] += 1  # A is younger than C
    else:
        ages['C'] += 1  # A is older than C

    if S_BC == '<':
        ages['B'] += 1  # B is younger than C
    else:
        ages['C'] += 1  # B is older than C

    # Sort brothers by their ages
    sorted_brothers = sorted(ages.items(), key=lambda item: item[1])
    
    # The middle brother is the second in the sorted order
    return sorted_brothers[1][0]

# Read input
import sys
input_data = sys.stdin.read().strip()
S_AB, S_AC, S_BC = input_data.split()

# Find and print the middle brother
middle_brother = find_middle_brother(S_AB, S_AC, S_BC)
print(middle_brother)