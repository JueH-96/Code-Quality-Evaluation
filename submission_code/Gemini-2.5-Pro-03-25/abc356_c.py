# YOUR CODE HERE
import sys

def solve():
    # Read N, M, K from the first line of input
    # N: number of keys
    # M: number of tests
    # K: minimum number of real keys required to open the door
    N, M, K = map(int, sys.stdin.readline().split())
    
    # List to store processed test data
    tests_data = []
    
    # Read M lines, each describing a test
    for _ in range(M):
        # Read the line and split it into parts
        line = sys.stdin.readline().split()
        
        # The first element is C_i, the number of keys used in the test. It is not explicitly needed
        # if we parse keys based on position relative to the result.
        # C_i = int(line[0]) 
        
        # The last element of the split line is the result ('o' for opened, 'x' for not opened)
        R_i = line[-1]
        
        # The elements from index 1 up to (but not including) the last element are the key numbers used in the test.
        # Key numbers are given 1-based, convert them to 0-based indices for bitmask operations.
        key_indices = [int(x) - 1 for x in line[1:-1]]
        
        # Create a bitmask representing the set of keys used in this test.
        # The k-th bit (0-indexed) is set to 1 if key k+1 (1-based) was used in this test.
        test_mask = 0
        for key_idx in key_indices:
            # Set the corresponding bit in the test_mask
             test_mask |= (1 << key_idx)
            
        # Store the precomputed test mask and the expected result for this test
        tests_data.append({'test_mask': test_mask, 'result': R_i})

    # Initialize a counter for the number of valid assignments of real/dummy status to keys
    valid_assignments_count = 0
    
    # Iterate through all 2^N possible assignments of keys.
    # Each assignment is represented by a bitmask `mask` of length N.
    # The `mask` ranges from 0 to 2^N - 1.
    # If the k-th bit (0-indexed) of `mask` is 1, key k+1 (1-based index) is considered real in this assignment.
    # If the k-th bit is 0, key k+1 is considered dummy.
    for mask in range(1 << N): 
        # Assume the current assignment `mask` is valid until a contradiction with any test result is found.
        is_valid_mask = True
        
        # Check the current assignment `mask` against all M recorded test results.
        for test in tests_data:
            test_mask = test['test_mask'] # Bitmask representing keys used in this test
            expected_result = test['result'] # Recorded result ('o' or 'x')
            
            # Determine the set of real keys that were used in this test according to the current assignment `mask`.
            # This is achieved by finding the intersection of the set of all real keys defined by `mask`
            # and the set of keys used in the test defined by `test_mask`.
            # The intersection is computed using the bitwise AND operation.
            intersection_mask = mask & test_mask
            
            # Count how many real keys were used in the test.
            # This is equivalent to counting the number of set bits (population count) in the `intersection_mask`.
            # The `bin(integer).count('1')` method efficiently counts set bits in Python.
            real_key_count = bin(intersection_mask).count('1')

            # Determine if the door *should* have opened based on the rule:
            # The door opens if and only if the number of real keys used (`real_key_count`) is at least K.
            door_opened = (real_key_count >= K)
            
            # Check if the outcome determined by the current assignment `mask` contradicts the recorded test result.
            if expected_result == 'o':
                # If the test recorded 'o' (opened), but our assignment implies it shouldn't have opened (`not door_opened`):
                if not door_opened:
                    is_valid_mask = False # Contradiction found. This assignment is invalid.
                    break # No need to check further tests for this mask; proceed to the next assignment mask.
            else: # expected_result == 'x'
                # If the test recorded 'x' (did not open), but our assignment implies it should have opened (`door_opened`):
                if door_opened:
                    is_valid_mask = False # Contradiction found. This assignment is invalid.
                    break # No need to check further tests for this mask; proceed to the next assignment mask.

        # If `is_valid_mask` remains True after checking against all M tests, it means this assignment `mask`
        # is consistent with all recorded test results.
        if is_valid_mask:
            valid_assignments_count += 1 # Increment the count of valid assignments.

    # After checking all 2^N possible assignments, print the total count of valid assignments found.
    print(valid_assignments_count)

# Call the solve function to execute the main logic of the program.
solve()