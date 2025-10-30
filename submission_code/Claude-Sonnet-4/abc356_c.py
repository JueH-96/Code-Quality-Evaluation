# YOUR CODE HERE
def solve():
    line = input().split()
    N, M, K = int(line[0]), int(line[1]), int(line[2])
    
    tests = []
    for _ in range(M):
        line = input().split()
        C = int(line[0])
        keys = [int(line[i]) for i in range(1, C + 1)]
        result = line[C + 1]
        tests.append((keys, result))
    
    count = 0
    
    # Try all 2^N combinations of real/dummy keys
    for mask in range(1 << N):
        # mask represents which keys are real (bit i = 1 means key i+1 is real)
        valid = True
        
        # Check each test
        for keys, result in tests:
            # Count how many real keys were used in this test
            real_keys_used = 0
            for key in keys:
                if mask & (1 << (key - 1)):  # key is real
                    real_keys_used += 1
            
            # Check if this matches the expected result
            if result == 'o':
                # Door opened, so we need at least K real keys
                if real_keys_used < K:
                    valid = False
                    break
            else:  # result == 'x'
                # Door didn't open, so we need fewer than K real keys
                if real_keys_used >= K:
                    valid = False
                    break
        
        if valid:
            count += 1
    
    print(count)

solve()