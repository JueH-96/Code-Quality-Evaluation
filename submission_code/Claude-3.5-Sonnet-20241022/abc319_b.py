N = int(input())

result = []
for i in range(N + 1):
    min_divisor = float('inf')
    for j in range(1, 10):
        if N % j == 0 and i % (N // j) == 0:
            min_divisor = min(min_divisor, j)
    
    if min_divisor == float('inf'):
        result.append('-')
    else:
        result.append(str(min_divisor))

print(''.join(result))