# YOUR CODE HERE
S = input().strip()
T = input().strip()

# Count characters in both strings
count_S = {}
count_T = {}

for c in S:
    count_S[c] = count_S.get(c, 0) + 1

for c in T:
    count_T[c] = count_T.get(c, 0) + 1

# Get counts of @ symbols
at_S = count_S.get('@', 0)
at_T = count_T.get('@', 0)

# Remove @ from counts as they will be replaced
if '@' in count_S:
    del count_S['@']
if '@' in count_T:
    del count_T['@']

# Check if we can balance non-@ characters
# We need to see if the difference in counts can be compensated by @ replacements
deficit_S = 0  # How many more characters S needs
deficit_T = 0  # How many more characters T needs

# Get all unique characters that appear in either string
all_chars = set(count_S.keys()) | set(count_T.keys())

for char in all_chars:
    s_count = count_S.get(char, 0)
    t_count = count_T.get(char, 0)
    
    if s_count > t_count:
        deficit_T += s_count - t_count
    elif t_count > s_count:
        deficit_S += t_count - s_count

# Check if we can satisfy the deficits with available @ symbols
# Also, @ symbols can only be replaced with letters from {a, t, c, o, d, e, r}
allowed_chars = set('atcoder')

# Check if all non-@ characters are from allowed set or if we have enough @ to compensate
for char in all_chars:
    if char not in allowed_chars:
        # This character cannot be created by replacing @
        # So both strings must have equal counts of this character
        if count_S.get(char, 0) != count_T.get(char, 0):
            print("No")
            exit()

# Recalculate deficits considering only allowed characters
deficit_S = 0
deficit_T = 0

for char in all_chars:
    if char in allowed_chars:
        s_count = count_S.get(char, 0)
        t_count = count_T.get(char, 0)
        
        if s_count > t_count:
            deficit_T += s_count - t_count
        elif t_count > s_count:
            deficit_S += t_count - s_count

# Check if we can satisfy deficits with @ symbols
if deficit_S <= at_S and deficit_T <= at_T:
    # Check if remaining @ symbols can be balanced
    remaining_at_S = at_S - deficit_S
    remaining_at_T = at_T - deficit_T
    
    # Remaining @ symbols must be equal (they can be replaced with same characters)
    if remaining_at_S == remaining_at_T:
        print("Yes")
    else:
        print("No")
else:
    print("No")