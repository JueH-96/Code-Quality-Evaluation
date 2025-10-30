def solve():
    # Read input
    N, M, K = map(int, input().strip().split())
    
    tests = []
    for _ in range(M):
        line = input().strip().split()
        C_i = int(line[0])
        keys_used = list(map(int, line[1:C_i+1]))
        result = line[C_i+1]
        tests.append((keys_used, result))
    
    # Count valid combinations
    valid_count = 0
    for config in range(2**N):
        # Check if all test results are consistent
        all_consistent = True
        for keys_used, result in tests:
            # Count real keys among the inserted ones
            real_count = 0
            for key in keys_used:
                # Check if the key is real in this configuration
                if (config >> (key-1)) & 1:
                    real_count += 1
            
            # Check consistency
            if result == 'o' and real_count < K:
                all_consistent = False
                break
            if result == 'x' and real_count >= K:
                all_consistent = False
                break
        
        # Increment count if all tests are consistent
        if all_consistent:
            valid_count += 1
    
    return valid_count

print(solve())