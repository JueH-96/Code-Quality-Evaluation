def can_form_same_string(S, X, Y):
    # Find the first occurrence of '1' in both X and Y
    first_one_in_x = X.find('1')
    first_one_in_y = Y.find('1')

    # If both X and Y do not contain '1', they are already equal
    if first_one_in_x == -1 and first_one_in_y == -1:
        return True

    # If only one of them contains '1', they cannot be equal
    if first_one_in_x == -1 or first_one_in_y == -1:
        return False

    # Check if the parts before the first '1' are of the same length
    if first_one_in_x != first_one_in_y:
        return False

    # Check if the parts after the first '1' are the same
    return X[first_one_in_x:] == Y[first_one_in_y:]

# Read the number of test cases
t = int(input().strip())

# Process each test case
for _ in range(t):
    S = input().strip()
    X = input().strip()
    Y = input().strip()

    # Output the result for each test case
    print("Yes" if can_form_same_string(S, X, Y) else "No")