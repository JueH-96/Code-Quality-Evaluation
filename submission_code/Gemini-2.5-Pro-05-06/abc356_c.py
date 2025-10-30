import sys

def solve():
    N, M, K = map(int, sys.stdin.readline().split())

    tests_data = []
    for _ in range(M):
        line_parts = sys.stdin.readline().split()
        
        C_val = int(line_parts[0])
        # Keys are from index 1 up to index C_val (inclusive) in line_parts
        key_list_1_indexed = [int(line_parts[j+1]) for j in range(C_val)]
        # Convert to 0-indexed for bitmask operations
        key_list_0_indexed = [k-1 for k in key_list_1_indexed] 
        
        # Result char is at index C_val + 1 in line_parts
        result_char = line_parts[C_val+1]
        
        tests_data.append({'keys': key_list_0_indexed, 'result_char': result_char})

    ans_count = 0

    # Iterate through all 2^N possible assignments of real/dummy keys
    # 'current_key_config_mask' is a bitmask: 
    # if k-th bit is set, key k (0-indexed) is real, else dummy.
    for current_key_config_mask in range(1 << N): # Iterates from 0 to 2^N - 1
        this_mask_is_valid = True # Assume initially this configuration is valid
        
        # Check this configuration against all M tests
        for test_info in tests_data:
            keys_used_in_test = test_info['keys']
            expected_result_char = test_info['result_char']

            num_real_keys_inserted_in_this_test = 0
            for key_0_idx in keys_used_in_test:
                # Check if this key (key_0_idx) is real in the current_key_config_mask
                if (current_key_config_mask >> key_0_idx) & 1: # if key_0_idx'th bit is set
                    num_real_keys_inserted_in_this_test += 1
            
            # Determine if Door X would open based on this configuration and keys inserted
            door_would_open = (num_real_keys_inserted_in_this_test >= K)
            
            # Determine if the test result indicated Door X opened
            test_reported_door_opened = (expected_result_char == 'o')

            # If the behavior of Door X for this configuration contradicts the test report
            if door_would_open != test_reported_door_opened:
                this_mask_is_valid = False # This configuration is contradictory
                break # No need to check further tests for this configuration, move to next mask
        
        if this_mask_is_valid: # If mask was consistent with all tests
            ans_count += 1
            
    sys.stdout.write(str(ans_count) + "
")

if __name__ == '__main__':
    solve()