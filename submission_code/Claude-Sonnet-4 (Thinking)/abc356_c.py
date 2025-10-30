n, m, k = map(int, input().split())

tests = []
for _ in range(m):
    line = input().split()
    c = int(line[0])
    keys = [int(line[i]) for i in range(1, c+1)]
    result = line[c+1]
    tests.append((keys, result))

count = 0

# Try all 2^n combinations
for mask in range(1 << n):
    # mask represents which keys are real (bit i = 1 means key i+1 is real)
    valid = True
    
    for keys, result in tests:
        # Count real keys used in this test
        real_count = 0
        for key in keys:
            if mask & (1 << (key - 1)):  # key is real
                real_count += 1
        
        # Check if this matches the expected result
        if result == 'o':
            if real_count < k:
                valid = False
                break
        else:  # result == 'x'
            if real_count >= k:
                valid = False
                break
    
    if valid:
        count += 1

print(count)