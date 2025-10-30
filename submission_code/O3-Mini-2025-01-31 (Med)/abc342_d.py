def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Define maximum value needed for sieve
    max_val = 200000
    
    # Precompute smallest prime factors (spf) for numbers up to max_val.
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def squarefree(x):
        # Returns the squarefree part of a positive integer x.
        # This is computed by removing all squared factors.
        result = 1
        while x > 1:
            p = spf[x]
            count = 0
            while x % p == 0:
                count += 1
                x //= p
            # Multiply by p only if its exponent is odd.
            if count % 2 == 1:
                result *= p
        return result

    # We need to treat zeros separately since 0 multiplied by any number is 0 (a perfect square).
    count_zero = 0
    freq = {}  # dictionary to count frequencies of squarefree parts for nonzero A_i's

    for num in A:
        if num == 0:
            count_zero += 1
        else:
            sf = squarefree(num)
            freq[sf] = freq.get(sf, 0) + 1

    # Now, if we write any nonzero integer A[i] as m^2 * s (where s is the squarefree part),
    # then A[i] * A[j] = (m_i*m_j)^2 * (s_i * s_j).
    # For the product to be a perfect square, we require s_i * s_j to be a perfect square.
    # Since both s_i and s_j are squarefree, this happens if and only if s_i == s_j.
    # Thus, among nonzero numbers, count pairs with the same squarefree part.
    res = 0
    for count in freq.values():
        res += count * (count - 1) // 2

    # And count the pairs where at least one number is zero.
    # Note: 0 is a perfect square, so any product involving 0 is a perfect square.
    nonzero_count = n - count_zero
    res += count_zero * nonzero_count           # each zero paired with nonzero
    res += count_zero * (count_zero - 1) // 2     # pairs of zeros

    sys.stdout.write(str(res))


if __name__ == '__main__':
    main()