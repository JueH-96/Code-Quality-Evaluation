def solve(N):
    result = []
    for i in range(N+1):
        if i % N == 0:
            result.append(1)
        else:
            divisors = [j for j in range(1, 10) if N % j == 0 and i % (N // j) == 0]
            if divisors:
                result.append(min(divisors))
            else:
                result.append(-1)
    return ''.join(map(str, result))

N = int(input())
print(solve(N))