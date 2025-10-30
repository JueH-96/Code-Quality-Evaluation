import sys
import threading

def main():
    import sys
    import math
    from bisect import bisect_right

    data = sys.stdin.read().strip().split()
    N = int(data[0])

    # We need primes up to floor(sqrt(N))
    S = math.isqrt(N)
    max_n = S

    # Sieve of Eratosthenes
    sieve = bytearray(b'\x01') * (max_n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(max_n**0.5) + 1):
        if sieve[i]:
            step = i
            start = i*i
            sieve[start:max_n+1:step] = b'\x00' * (((max_n - start)//step) + 1)

    primes = [i for i, is_p in enumerate(sieve) if is_p]

    # Count numbers of form p^8 <= N
    cnt = 0
    # p^8 grows fast; only very small p
    for p in primes:
        # since primes list starts from 2, break if p^8 > N
        if p**8 > N:
            break
        cnt += 1

    # Count numbers of form p^2 * q^2 <= N with p < q
    # We want p * q <= sqrt(N), i.e. q <= sqrt(N) // p
    # Let sqrtN = floor(sqrt(N))
    sqrtN = S
    L = len(primes)
    for i, p in enumerate(primes):
        # minimal q is primes[i+1], so if that already fails, break
        if i+1 >= L:
            break
        min_q = primes[i+1]
        if p * min_q > sqrtN:
            break

        limit_q = sqrtN // p
        # find rightmost index of prime <= limit_q
        j = bisect_right(primes, limit_q)  # gives insertion point
        # valid q indices are i+1 .. j-1
        if j > i+1:
            cnt += (j - (i+1))

    print(cnt)

if __name__ == "__main__":
    main()