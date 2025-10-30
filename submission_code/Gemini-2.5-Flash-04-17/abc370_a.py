# Read the input line
line = input()

# Split the line into two strings
l_str, r_str = line.split()

# Convert the strings to integers
L = int(l_str)
R = int(r_str)

# Apply the logic based on L and R
# If only left hand is raised (L=1, R=0), he wants takoyaki (Yes)
if L == 1 and R == 0:
    print("Yes")
# If only right hand is raised (L=0, R=1), he doesn't want takoyaki (No)
elif L == 0 and R == 1:
    print("No")
# If both hands are raised (L=1, R=1) or no hands are raised (L=0, R=0), it's invalid
else:
    print("Invalid")