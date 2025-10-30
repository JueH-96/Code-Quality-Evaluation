import sys
import threading

def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])

    # For N < 4, the only perfect power in [1..N] is 1 = 1^b, b>=2.
    if N < 4:
        print(1)
        return

    # threshold = floor(log2(N))
    threshold = N.bit_length() - 1

    # get all primes up to threshold (threshold <= ~60)
    def primes_upto(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    primes = primes_upto(threshold)
    p = len(primes)

    # Precompute mapping from single-bit mask to prime index
    lowbit_idx = {1 << i: i for i in range(p)}

    size = 1 << p
    # dp_lcm[mask] = product of primes in the subset 'mask'
    dp_lcm = [1] * size
    # dp_parity[mask] = parity of popcount(mask): 1 if odd size, 0 if even size
    dp_parity = [0] * size

    # integer k-th root: floor(n^(1/k))
    def kth_root_floor(n, k):
        # returns the largest x such that x^k <= n
        if k == 1:
            return n
        # bit_length of n
        bl = n.bit_length()
        # high bound = 2^h where h = floor((bl-1)/k) + 2
        # then high^k > n, so search in [1..high)
        h = (bl - 1) // k + 2
        high = 1 << h
        low = 1
        # binary search for the boundary
        while low < high:
            mid = (low + high) // 2
            # mid^k might be large but Python handles big ints
            if pow(mid, k) <= n:
                low = mid + 1
            else:
                high = mid
        # low is first x with x^k > n, so answer is low-1
        return low - 1

    ans = 0
    # iterate non-empty subsets of primes by bitmask
    for mask in range(1, size):
        lb = mask & -mask
        i = lowbit_idx[lb]
        prev = mask ^ lb

        # update lcm = product of primes in the subset
        dp_lcm[mask] = dp_lcm[prev] * primes[i]
        # update parity bit (popcount % 2)
        dp_parity[mask] = dp_parity[prev] ^ 1

        # inclusion-exclusion sign: +1 if subset size odd, -1 if even
        sign = 1 if dp_parity[mask] == 1 else -1

        k = dp_lcm[mask]
        # if k > threshold, then N^(1/k) < 2 => floor = 1
        if k > threshold:
            cnt = 1
        else:
            cnt = kth_root_floor(N, k)

        ans += sign * cnt

    # print result
    print(ans)

if __name__ == "__main__":
    main()