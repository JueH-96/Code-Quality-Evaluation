# YOUR CODE HERE
N = int(input())

result = []

for i in range(N + 1):
    smallest_j = None
    
    # Check divisors from 1 to 9
    for j in range(1, 10):
        if N % j == 0:  # j is a divisor of N
            if i % (N // j) == 0:  # i is a multiple of N/j
                smallest_j = j
                break  # We found the smallest, so break
    
    if smallest_j is not None:
        result.append(str(smallest_j))
    else:
        result.append('-')

print(''.join(result))