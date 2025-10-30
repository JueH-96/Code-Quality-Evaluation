# YOUR CODE HERE
import sys

def solve():
    """
    Reads the problem input, finds the number of valid key combinations,
    and prints the result.
    """
    try:
        # Read problem parameters N, M, K
        # N: total number of keys
        # M: number of tests
        # K: number of real keys required to open the door
        line = sys.stdin.readline()
        if not line: return
        n, m, k = map(int, line.split())
    except (IOError, ValueError):
        # Handle cases with malformed or empty input.
        return

    # Store pre-processed test data.
    # Each test is a tuple: (bitmask_of_keys_used, result_character)
    tests = []
    for _ in range(m):
        line = sys.stdin.readline().split()
        if not line: continue
        
        # The test result ('o' or 'x') is the last element.
        result = line[-1]
        
        # The keys used in the test are the integers between the first and last elements.
        keys_used = [int(x) for x in line[1:-1]]
        
        # Convert the list of keys into a bitmask for efficient operations.
        # A key with number `j` corresponds to the (j-1)-th bit.
        test_mask = 0
        for key_num in keys_used:
            test_mask |= (1 << (key_num - 1))
        
        tests.append((test_mask, result))

    # This counter will store the number of valid key combinations.
    valid_combinations_count = 0

    # Iterate through all 2^N possible combinations of which keys are real.
    # We use a bitmask 'i' to represent a combination.
    # If the j-th bit of 'i' is 1, key j+1 is real. Otherwise, it's a dummy.
    for i in range(1 << n):
        # Assume the current combination 'i' is valid until a contradiction is found.
        is_combination_consistent = True
        
        # Check this combination against all M test results.
        for test_mask, expected_result in tests:
            # Find which of the supposedly real keys (from mask `i`) were used in this test.
            real_keys_in_test_mask = i & test_mask
            
            # Count the number of set bits (popcount) to get the number of real keys.
            num_real_keys = bin(real_keys_in_test_mask).count('1')
            
            # Check for contradiction with the test result.
            if expected_result == 'o':
                # If the door opened, there must be at least K real keys.
                if num_real_keys < k:
                    is_combination_consistent = False
                    break
            else:  # expected_result == 'x'
                # If the door did not open, there must be fewer than K real keys.
                if num_real_keys >= k:
                    is_combination_consistent = False
                    break
        
        # If no contradictions were found after all tests, this combination is valid.
        if is_combination_consistent:
            valid_combinations_count += 1
            
    # Print the total count of valid combinations.
    print(valid_combinations_count)

solve()