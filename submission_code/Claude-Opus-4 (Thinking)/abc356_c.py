# YOUR CODE HERE
n, m, k = map(int, input().split())
tests = []
for _ in range(m):
    parts = input().split()
    c = int(parts[0])
    keys = [int(parts[i]) for i in range(1, c + 1)]
    result = parts[c + 1]
    tests.append((keys, result))

count = 0
# Try all 2^n combinations
for mask in range(1 << n):
    valid = True
    # Check if this combination satisfies all tests
    for keys, result in tests:
        real_keys = sum(1 for key in keys if mask & (1 << (key - 1)))
        
        if result == 'o' and real_keys < k:
            valid = False
            break
        elif result == 'x' and real_keys >= k:
            valid = False
            break
    
    if valid:
        count += 1

print(count)