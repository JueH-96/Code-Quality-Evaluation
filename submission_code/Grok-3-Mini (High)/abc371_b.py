import sys

# Read all input from stdin and split into a list
data = sys.stdin.read().split()

# Initialize index to 0
index = 0

# Read N and M
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Initialize a list to keep track of whether a male has been born in each family
male_born = [False] * N

# Process each baby
for i in range(M):
    # Read A and B for the current baby
    A_str = data[index]
    index += 1
    B_str = data[index]
    index += 1
    
    # Convert A to integer
    A = int(A_str)
    B = B_str  # B is 'M' or 'F'
    
    # Check if the baby is female
    if B == 'F':
        print("No")
    else:  # Baby is male
        # Get the 0-based family index
        fam_index = A - 1
        # Check if this is the first male in the family
        if not male_born[fam_index]:
            male_born[fam_index] = True
            print("Yes")
        else:
            print("No")