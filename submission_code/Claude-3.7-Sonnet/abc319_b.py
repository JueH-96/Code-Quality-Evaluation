def solve(N):
    result = []
    for i in range(N + 1):
        smallest_j = None
        for j in range(1, 10):  # Check divisors from 1 to 9
            if N % j == 0:  # If j is a divisor of N
                if i % (N // j) == 0:  # If i is a multiple of N/j
                    smallest_j = j
                    break
        if smallest_j is not None:
            result.append(str(smallest_j))
        else:
            result.append('-')
    return ''.join(result)

N = int(input())
print(solve(N))