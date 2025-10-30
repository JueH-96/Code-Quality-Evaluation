import sys
import threading

def main():
    import sys
    import math
    import bisect

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])

    # We need primes up to sqrt(N)
    max_sieve = math.isqrt(N)
    if max_sieve < 2:
        print(0)
        return

    # Sieve of Eratosthenes up to max_sieve
    sieve = bytearray(b'\x01') * (max_sieve + 1)
    sieve[0:2] = b'\x00\x00'
    lim = int(max_sieve**0.5) + 1
    for i in range(2, lim):
        if sieve[i]:
            step = i
            start = i*i
            sieve[start:max_sieve+1:step] = b'\x00' * (((max_sieve - start)//step) + 1)
    primes = [i for i, is_pr in enumerate(sieve) if is_pr]

    # Count numbers of form p^8 <= N
    cnt_p8 = 0
    for p in primes:
        # once p^8 exceeds N, no further primes will satisfy
        if p**8 > N:
            break
        cnt_p8 += 1

    # Count numbers of form p^2 * q^2 <= N, p < q
    # That is p*q <= sqrt(N)
    M = math.isqrt(N)
    cnt_pairs = 0
    # We only need p up to sqrt(M), since p < q and p*q <= M => p < sqrt(M)
    # So break when p*p > M
    for i, p in enumerate(primes):
        if p * p > M:
            break
        # maximum q is M // p
        max_q = M // p
        # find number of primes <= max_q
        # bisect_right gives index of first > max_q
        j = bisect.bisect_right(primes, max_q)
        # count of primes > p and <= max_q is j - (i+1)
        cnt = j - (i + 1)
        if cnt > 0:
            cnt_pairs += cnt

    # Total answer
    print(cnt_p8 + cnt_pairs)

# run main
if __name__ == "__main__":
    main()