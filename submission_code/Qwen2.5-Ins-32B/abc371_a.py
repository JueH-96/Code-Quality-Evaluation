import sys

input = sys.stdin.read

def find_middle_brother(s_ab, s_ac, s_bc):
    # Initialize the ages of A, B, and C as None
    ages = {'A': None, 'B': None, 'C': None}
    
    # Determine the relative ages based on the input strings
    if s_ab == '<':
        ages['A'] = 0
        ages['B'] = 1
    else:
        ages['A'] = 1
        ages['B'] = 0
    
    if s_ac == '<':
        if ages['A'] is None:
            ages['A'] = 0
            ages['C'] = 1
        elif ages['A'] == 0:
            ages['C'] = 1
        else:
            ages['C'] = 2
    else:
        if ages['A'] is None:
            ages['A'] = 1
            ages['C'] = 0
        elif ages['A'] == 1:
            ages['C'] = 0
        else:
            ages['C'] = 2
    
    if s_bc == '<':
        if ages['B'] is None:
            ages['B'] = 0
            ages['C'] = 1
        elif ages['B'] == 0:
            ages['C'] = 1
        else:
            ages['C'] = 2
    else:
        if ages['B'] is None:
            ages['B'] = 1
            ages['C'] = 0
        elif ages['B'] == 1:
            ages['C'] = 0
        else:
            ages['C'] = 2
    
    # Find the middle brother
    middle_brother = max(ages, key=ages.get) if list(ages.values()).count(1) == 1 else min(ages, key=ages.get)
    return middle_brother

# Read the input
s_ab, s_ac, s_bc = input().strip().split()

# Find and print the middle brother
print(find_middle_brother(s_ab, s_ac, s_bc))