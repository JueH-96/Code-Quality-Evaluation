# Read N and Q from the first line of input
N, Q = map(int, input().split())

# Read the list of treatments T_1, T_2, ..., T_Q from the second line
treatments = list(map(int, input().split()))

# Initialize the state of the teeth.
# We use a list `has_tooth` of N booleans.
# `has_tooth[i]` is True if hole (i+1) has a tooth, and False otherwise.
# Hole numbers are 1-indexed (1 to N). List indices are 0-indexed (0 to N-1).
# Initially, Takahashi has N teeth, one in each hole.
has_tooth = [True] * N 

# Process each treatment
for t_val in treatments:
    # Convert the 1-based hole number T_i to a 0-based list index
    hole_index = t_val - 1
    
    # Apply the treatment:
    # - If there is a tooth (has_tooth[hole_index] is True), remove it (set to False).
    # - If there is no tooth (has_tooth[hole_index] is False), grow one (set to True).
    # This is equivalent to flipping the boolean value.
    has_tooth[hole_index] = not has_tooth[hole_index]

# After all treatments, count the number of teeth.
# In Python, when summing a list of booleans, True is treated as 1 and False as 0.
final_teeth_count = sum(has_tooth)

# Print the final count of teeth
print(final_teeth_count)