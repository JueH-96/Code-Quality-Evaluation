N = int(input())

result = []

for i in range(N + 1):
    valid_divisors = []
    
    # Check all possible j from 1 to 9
    for j in range(1, 10):
        # Check if j is a divisor of N
        if N % j == 0:
            # Check if i is a multiple of N/j
            if i % (N // j) == 0:
                valid_divisors.append(j)
    
    if valid_divisors:
        result.append(str(min(valid_divisors)))
    else:
        result.append('-')

print(''.join(result))