def count_nine_divisors(n):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    count = 0
    for i in range(1 << len(primes)):
        p1 = 1
        p2 = 1
        for j in range(len(primes)):
            if (i >> j) & 1:
                if p1 == 1:
                    p1 = primes[j]
                else:
                    p2 = primes[j]
        if p2 == 1:
            x = p1**8
            if x <= n:
                count += 1
        else:
            x = p1**2 * p2**2
            if x <= n:
                count += 1
    return count

n = int(input())
print(count_nine_divisors(n))