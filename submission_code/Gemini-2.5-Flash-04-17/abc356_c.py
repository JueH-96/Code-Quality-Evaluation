# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read N, M, K from the first line
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])
    K = int(line1[2])

    # Read the M test cases
    tests = []
    for _ in range(M):
        line = sys.stdin.readline().split()
        C_i = int(line[0])
        # Read the keys inserted in the test (1-indexed)
        # line[1 : C_i + 1] gets the key strings, map(int, ...) converts them to integers
        keys_in_test = list(map(int, line[1 : C_i + 1]))
        # Read the test result ('o' or 'x')
        result_expected = line[C_i + 1]
        # Store the list of keys and the expected result for this test
        tests.append((keys_in_test, result_expected))

    # Initialize count of valid assignments
    valid_assignments_count = 0

    # Iterate through all possible assignments of real/dummy keys.
    # There are 2^N possibilities. Each integer 'mask' from 0 to 2^N - 1
    # represents an assignment.
    # If the i-th bit (0-indexed) of 'mask' is 1, it means key i+1 is real.
    # If the i-th bit is 0, key i+1 is dummy.
    for mask in range(2**N):
        # Assume the current assignment is valid until proven otherwise
        is_valid_assignment = True

        # Check this assignment against all M test results
        for keys_in_test, result_expected in tests:
            real_count = 0
            # Count how many of the keys used in the current test are real
            # according to the current assignment (mask)
            for key in keys_in_test:
                # Check if the (key-1)-th bit of mask is set (key is 1-indexed)
                # (mask >> (key - 1)) shifts the bit for key-1 to the 0th position
                # & 1 checks if that bit is 1
                if (mask >> (key - 1)) & 1:
                    real_count += 1

            # Check if the calculated real count is consistent with the test result and K
            if result_expected == 'o':
                # Door opened ('o') requires at least K real keys
                if real_count < K:
                    # This assignment contradicts the test result
                    is_valid_assignment = False
                    # No need to check other tests for this assignment, it's already invalid
                    break
            elif result_expected == 'x':
                # Door did not open ('x') requires less than K real keys
                if real_count >= K:
                    # This assignment contradicts the test result
                    is_valid_assignment = False
                    # No need to check other tests for this assignment, it's already invalid
                    break

        # If the assignment was consistent with all tests
        if is_valid_assignment:
            # Increment the count of valid assignments
            valid_assignments_count += 1

    # Print the final count of valid assignments
    print(valid_assignments_count)

# Execute the solve function
solve()