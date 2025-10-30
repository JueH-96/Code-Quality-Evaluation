import math

def sieve(n):
    is_prime = [True] * (n+1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n+1) if is_prime[i]]
    return primes

def exponent_in_factorial(n, p):
    count = 0
    power = p
    while power <= n:
        count += n // power
        power *= p
    return count

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    L_total = N * K
    if L_total == 0:
        print("")
        return

    primes = sieve(L_total)
    
    if N == 1 and K == 1:
        print("1")
        return

    root_count = 1
    for p in primes:
        exp1 = exponent_in_factorial(L_total, p)
        exp2 = exponent_in_factorial(K, p)
        exp_val = exp1 - N * exp2
        if exp_val > 0:
            term = pow(p, exp_val)
            root_count = root_count * term

    current_rank = (root_count + 1) // 2
    counts = [K] * (N + 1)
    L = L_total
    current_count = root_count
    result = []

    for _ in range(L_total):
        found = False
        for i in range(1, N + 1):
            if counts[i] == 0:
                continue
            count_i = current_count * counts[i] // L
            if count_i < current_rank:
                current_rank -= count_i
            else:
                counts[i] -= 1
                result.append(i)
                current_count = count_i
                L -= 1
                found = True
                break
        if not found:
            for i in range(1, N + 1):
                if counts[i] > 0:
                    counts[i] -= 1
                    result.append(i)
                    L -= 1
                    break

    print(" ".join(map(str, result)))

if __name__ == '__main__':
    main()