# YOUR CODE HERE

# Read the inputs from stdin
input_str = input()

# Split the input string into a list of characters
input_list = input_str.split()

# Initialize a dictionary to store the age relationships
age_relationships = {'A': 0, 'B': 0, 'C': 0}

# Update the age relationships based on the input
if input_list[0] == '<':
    age_relationships['A'] += 1
    age_relationships['B'] -= 1
else:
    age_relationships['A'] -= 1
    age_relationships['B'] += 1

if input_list[1] == '<':
    age_relationships['A'] += 1
    age_relationships['C'] -= 1
else:
    age_relationships['A'] -= 1
    age_relationships['C'] += 1

if input_list[2] == '<':
    age_relationships['B'] += 1
    age_relationships['C'] -= 1
else:
    age_relationships['B'] -= 1
    age_relationships['C'] += 1

# Find the brother with the age relationship of 0
for brother, age in age_relationships.items():
    if age == 0:
        print(brother)
        break