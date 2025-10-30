import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Count zeros
    z = A.count(0)

    # Prepare sieve of smallest prime factors up to max(A)
    maxA = max(A)
    spf = [0] * (maxA + 1)
    for i in range(2, maxA + 1):
        if spf[i] == 0:
            for j in range(i, maxA + 1, i):
                if spf[j] == 0:
                    spf[j] = i

    from collections import Counter
    mask_count = Counter()

    # Function to compute the squarefree mask of a positive integer
    def squarefree_mask(x):
        # Factorize x using spf, collect primes with odd exponent
        last_p = 0
        cnt = 0
        mask_primes = []
        while x > 1:
            p = spf[x]
            x //= p
            if p == last_p:
                cnt += 1
            else:
                if last_p != 0 and (cnt & 1):
                    mask_primes.append(last_p)
                last_p = p
                cnt = 1
        # handle the final prime
        if last_p != 0 and (cnt & 1):
            mask_primes.append(last_p)
        # return as a tuple (immutable) for hashing
        if not mask_primes:
            return ()  # empty tuple for 1
        return tuple(mask_primes)

    # Build counts for non-zero values
    for x in A:
        if x != 0:
            m = squarefree_mask(x)
            mask_count[m] += 1

    # Count pairs among non-zeros with same mask
    ans = 0
    for cnt in mask_count.values():
        ans += cnt * (cnt - 1) // 2

    # Count pairs involving zeros: any pair with at least one zero gives product 0, a square
    # Pairs (zero, non-zero): z * (N - z)
    # Pairs (zero, zero): C(z,2)
    ans += z * (N - z) + z * (z - 1) // 2

    print(ans)

if __name__ == "__main__":
    main()