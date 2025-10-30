N = int(input())

# Find all divisors j of N where 1 <= j <= 9
divisors = [j for j in range(1, 10) if N % j == 0]

result = []
for i in range(N + 1):
    candidates = []
    for j in divisors:
        nj = N // j
        if i % nj == 0:
            candidates.append(j)
    if candidates:
        smallest_j = min(candidates)
        result.append(str(smallest_j))
    else:
        result.append('-')

print(''.join(result))