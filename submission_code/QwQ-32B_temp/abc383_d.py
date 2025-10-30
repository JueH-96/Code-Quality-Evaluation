import bisect
import math

def sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    primes = []
    for i in range(n+1):
        if is_prime[i]:
            primes.append(i)
    return primes

def main():
    N = int(input().strip())
    if N < 36:
        print(0)
        return

    S = int(math.isqrt(N))
    primes = sieve(S)
    
    # Compute count_A: primes p where p^8 <= N
    max_p_A = 1
    while True:
        next_p = max_p_A + 1
        if next_p ** 8 <= N:
            max_p_A = next_p
        else:
            break
    idx_A = bisect.bisect_right(primes, max_p_A)
    count_A = idx_A

    # Compute count_B: pairs p < q with p*q <= S
    count_B = 0
    len_primes = len(primes)
    for i in range(len_primes):
        p = primes[i]
        max_q_val = S // p
        low = i + 1
        if low >= len_primes:
            continue
        idx = bisect.bisect_right(primes, max_q_val, low, len_primes)
        count_B += (idx - low)
    
    total = count_A + count_B
    print(total)

if __name__ == "__main__":
    main()