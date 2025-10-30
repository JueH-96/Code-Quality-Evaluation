def main():
    import sys, math
    from collections import defaultdict

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    arr = [int(next(it)) for _ in range(n)]

    # Maximum value in the array (problem guarantees <= 200000)
    maxA = 200000
    
    # Precompute smallest prime factor (spf) for numbers up to maxA using sieve
    spf = list(range(maxA + 1))
    for i in range(2, int(maxA**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, maxA + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Function to compute the squarefree part of a positive integer.
    # Every number a can be expressed as (squarefree_part)*(perfect square).
    # For product A_i * A_j to be a square, for both nonzero numbers it is necessary
    # that they have the same squarefree part.
    def squarefree(x):
        res = 1
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                count += 1
                x //= p
            # If the exponent is odd, multiply p into the result.
            if count % 2 == 1:
                res *= p
        return res

    # Count frequencies:
    # For zeros: note that 0 is considered a square (0 = 0^2),
    # and 0 multiplied by any number is 0.
    cnt_zero = 0
    freq = defaultdict(int)
    for a in arr:
        if a == 0:
            cnt_zero += 1
        else:
            key = squarefree(a)
            freq[key] += 1

    # Count valid pairs:
    total_pairs = 0

    # 1. Pairs where at least one number is zero:
    # Every pair (0, x) yields product 0, which is a square.
    # There are pairs where one is zero and one is nonzero plus pairs of zeros.
    total_pairs += cnt_zero * (n - cnt_zero)          # one zero, one nonzero
    total_pairs += cnt_zero * (cnt_zero - 1) // 2       # both zeros

    # 2. Pairs of nonzero numbers such that their squarefree parts are equal.
    for count in freq.values():
        total_pairs += count * (count - 1) // 2

    sys.stdout.write(str(total_pairs))


if __name__ == "__main__":
    main()