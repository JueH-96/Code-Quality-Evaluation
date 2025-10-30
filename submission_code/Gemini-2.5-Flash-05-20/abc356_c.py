import sys

def solve():
    # Read N, M, K from the first line
    N, M, K = map(int, sys.stdin.readline().split())

    tests = []
    # Read M test cases
    for _ in range(M):
        line_parts = sys.stdin.readline().split()
        C_i = int(line_parts[0])
        # Keys are 1-indexed in input. We store them as is and adjust for 0-indexed bit manipulation.
        keys_in_test = [int(x) for x in line_parts[1:C_i+1]]
        result = line_parts[C_i+1]
        tests.append((keys_in_test, result))

    valid_combinations_count = 0

    # Iterate through all 2^N possible combinations of real/dummy keys.
    # Each integer 'i' represents a combination.
    # If the k-th bit (0-indexed) of 'i' is set, it means key (k+1) is real.
    # (1 << N) is equivalent to 2^N.
    for i in range(1 << N):
        is_consistent_combination = True

        # Check this combination against all M tests
        for test_keys, test_result in tests:
            real_keys_in_current_test_set = 0
            for key in test_keys:
                # Check if key (1-indexed) is real in the current combination 'i'.
                # Key 'key' corresponds to bit (key - 1).
                if (i >> (key - 1)) & 1: # If the (key-1)-th bit is 1, the key is real
                    real_keys_in_current_test_set += 1

            # Determine if this test is consistent with the current combination
            if test_result == 'o':
                # Door opened: must have at least K real keys
                if real_keys_in_current_test_set < K:
                    is_consistent_combination = False
                    break # This combination is inconsistent, no need to check further tests
            else: # test_result == 'x'
                # Door did not open: must have less than K real keys
                if real_keys_in_current_test_set >= K:
                    is_consistent_combination = False
                    break # This combination is inconsistent, no need to check further tests
        
        # If all tests were consistent for this combination, count it
        if is_consistent_combination:
            valid_combinations_count += 1

    # Print the total number of valid combinations
    print(valid_combinations_count)

# Call the solve function to run the program
solve()