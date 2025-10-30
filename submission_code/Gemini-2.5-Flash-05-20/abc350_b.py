# Read N (number of teeth/holes) and Q (number of treatments)
N, Q = map(int, input().split())

# Read the list of treatment hole IDs (T_i)
T = list(map(int, input().split()))

# Initialize a list to represent the state of each hole.
# We use 1-based indexing for holes, so the list size is N+1.
# has_tooth[i] will be True if hole i has a tooth, False otherwise.
# Initially, all N holes have a tooth.
has_tooth = [True] * (N + 1)

# Process each treatment
for hole_id in T:
    # The rule is:
    # - If there is a tooth in hole T_i, remove the tooth (set to False).
    # - If there is no tooth in hole T_i, grow a tooth (set to True).
    # This behavior is equivalent to flipping the boolean state of has_tooth[hole_id].
    has_tooth[hole_id] = not has_tooth[hole_id]

# After all treatments, count the number of teeth Takahashi has.
# We sum the boolean values from index 1 to N. In Python, True evaluates to 1 and False to 0.
# The element at index 0 (has_tooth[0]) is unused because holes are numbered 1 to N.
final_teeth_count = sum(has_tooth[1:])

# Print the final count
print(final_teeth_count)