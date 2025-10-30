# YOUR CODE HERE
# Read the input
S = input().strip()
T = input().strip()

# Define the positions of the points on the pentagon
# Assuming the pentagon is labeled A, B, C, D, E in order
# We can represent the positions as 0, 1, 2, 3, 4 respectively
point_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

# Get the indices of the points
S1_idx = point_to_index[S[0]]
S2_idx = point_to_index[S[1]]
T1_idx = point_to_index[T[0]]
T2_idx = point_to_index[T[1]]

# Calculate the distances
# Since it's a regular pentagon, the distance between two points is the minimum of the clockwise and counter-clockwise steps
def get_distance(idx1, idx2):
    diff = abs(idx1 - idx2)
    return min(diff, 5 - diff)

S_distance = get_distance(S1_idx, S2_idx)
T_distance = get_distance(T1_idx, T2_idx)

# Compare the distances
if S_distance == T_distance:
    print("Yes")
else:
    print("No")