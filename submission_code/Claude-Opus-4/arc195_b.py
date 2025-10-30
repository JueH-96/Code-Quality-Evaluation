# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Separate indices by type
fixed_both = []  # indices where both a[i] and b[i] are not -1
fixed_a = []     # indices where only a[i] is not -1
fixed_b = []     # indices where only b[i] is not -1
free_both = []   # indices where both are -1

# Also collect the actual values
a_values = []    # non-negative values from A
b_fixed = []     # (index, value) pairs for fixed B values

for i in range(n):
    if a[i] != -1 and b[i] != -1:
        fixed_both.append(i)
        a_values.append(a[i])
    elif a[i] != -1 and b[i] == -1:
        fixed_a.append(i)
        a_values.append(a[i])
    elif a[i] == -1 and b[i] != -1:
        fixed_b.append(i)
        b_fixed.append((i, b[i]))
    else:
        free_both.append(i)

# Check if any fixed values are negative
for i in range(n):
    if (a[i] != -1 and a[i] < 0) or (b[i] != -1 and b[i] < 0):
        print("No")
        exit()

# If there are positions where both are fixed, check if their sums are equal
if fixed_both:
    target_sum = a[fixed_both[0]] + b[fixed_both[0]]
    for i in fixed_both:
        if a[i] + b[i] != target_sum:
            print("No")
            exit()
else:
    # If no fixed sums, we need to determine a valid target sum
    # We need to ensure all sums can be made equal and non-negative
    target_sum = None

# Sort a_values in descending order for optimal pairing
a_values.sort(reverse=True)

# We need to assign A values to positions
# Priority: fixed_both positions (already assigned), then fixed_b positions

# For fixed_b positions, we need to check if we can assign A values
# such that A + B >= 0 and equals target_sum (if exists)

if fixed_both:
    # Remove already used A values for fixed_both positions
    used_a = []
    for i in fixed_both:
        used_a.append(a[i])
    
    # Remove used values from a_values
    for val in used_a:
        if val in a_values:
            a_values.remove(val)
    
    # Check if we can satisfy fixed_b positions
    b_fixed.sort(key=lambda x: x[1], reverse=True)  # Sort by B value descending
    
    for idx, b_val in b_fixed:
        needed_a = target_sum - b_val
        if needed_a < 0:
            print("No")
            exit()
        
        # Find if we have an A value that works
        found = False
        for i, a_val in enumerate(a_values):
            if a_val == needed_a:
                a_values.pop(i)
                found = True
                break
        
        if not found:
            # Check if we have free positions to create this value
            if not free_both and len(fixed_a) == 0:
                print("No")
                exit()
            # We'll need to use a free position or fixed_a position
            if len(a_values) > 0:
                a_values.pop(0)  # Use any available value
            elif free_both:
                free_both.pop()  # Use a free position
            else:
                print("No")
                exit()
else:
    # No fixed target sum, we have more flexibility
    # We need to ensure we can make all positions have equal non-negative sums
    pass

print("Yes")