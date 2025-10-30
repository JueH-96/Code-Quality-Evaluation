# Define the sets for sides and diagonals
sides = {('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B'), ('C', 'D'), 
         ('D', 'C'), ('D', 'E'), ('E', 'D'), ('E', 'A'), ('A', 'E')}
diagonals = {('A', 'C'), ('C', 'A'), ('B', 'D'), ('D', 'B'), 
             ('C', 'E'), ('E', 'C'), ('D', 'A'), ('A', 'D'), 
             ('E', 'B'), ('B', 'E')}

# Read input pairs
S1_S2 = input()
T1_T2 = input()

# Extract individual points
S1, S2 = S1_S2[0], S1_S2[1]
T1, T2 = T1_T2[0], T1_T2[1]

# Determine the type of each pair
S_pair = (S1, S2)
T_pair = (T1, T2)

# Check if both pairs are sides or both are diagonals
if (S_pair in sides and T_pair in sides) or (S_pair in diagonals and T_pair in diagonals):
    print("Yes")
else:
    print("No")