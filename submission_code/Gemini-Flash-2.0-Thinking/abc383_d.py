def solve():
    n = int(input())
    count = 0

    # Case 1: n = p^8
    p = 2
    while True:
        power = p**8
        if power > n:
            break
        count += 1
        p += 1

    # Case 2: n = p1^2 * p2^2
    sqrt_n = int(n**0.5)
    limit = sqrt_n + 1
    primes = []
    is_prime = [True] * limit
    for i in range(2, limit):
        if is_prime[i]:
            primes.append(i)
            for multiple in range(i * i, limit, i):
                is_prime[multiple] = False

    for i in range(len(primes)):
        p1 = primes[i]
        upper_bound = sqrt_n // p1
        for j in range(i + 1, len(primes)):
            p2 = primes[j]
            if p2 <= upper_bound:
                count += 1
            else:
                break

    print(count)

solve()