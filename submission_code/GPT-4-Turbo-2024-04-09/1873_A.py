def can_form_abc_with_one_swap(s):
    # Target string
    target = "abc"
    
    # If the string is already "abc"
    if s == target:
        return "YES"
    
    # Check if one swap can make it "abc"
    # We need to check if swapping any two characters can result in "abc"
    # There are only three possible swaps in a three-character string:
    # Swap 0 and 1: indices (0, 1)
    # Swap 0 and 2: indices (0, 2)
    # Swap 1 and 2: indices (1, 2)
    if (s[1] + s[0] + s[2] == target or
        s[2] + s[1] + s[0] == target or
        s[0] + s[2] + s[1] == target):
        return "YES"
    
    return "NO"

# Read number of test cases
t = int(input().strip())

# Process each test case
results = []
for _ in range(t):
    s = input().strip()
    result = can_form_abc_with_one_swap(s)
    results.append(result)

# Print results for each test case
for result in results:
    print(result)