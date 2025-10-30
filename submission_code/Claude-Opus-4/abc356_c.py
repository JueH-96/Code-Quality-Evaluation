# YOUR CODE HERE
N, M, K = map(int, input().split())

tests = []
for _ in range(M):
    line = input().split()
    C = int(line[0])
    keys = [int(line[i]) for i in range(1, C + 1)]
    result = line[C + 1]
    tests.append((keys, result))

count = 0

# Try all 2^N combinations
for mask in range(1 << N):
    valid = True
    
    # Check each test
    for keys, result in tests:
        # Count real keys in this test
        real_keys_count = 0
        for key in keys:
            if mask & (1 << (key - 1)):
                real_keys_count += 1
        
        # Check if this matches the test result
        if result == 'o':
            if real_keys_count < K:
                valid = False
                break
        else:  # result == 'x'
            if real_keys_count >= K:
                valid = False
                break
    
    if valid:
        count += 1

print(count)