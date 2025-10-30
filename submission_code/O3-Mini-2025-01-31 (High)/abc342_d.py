def main():
    import sys, math
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    
    # Count zeros and separate nonzero values.
    zero_count = 0
    nonzero = []
    for a in A:
        if a == 0:
            zero_count += 1
        else:
            nonzero.append(a)
    
    # Pairs that involve at least one zero.
    # Since 0 is a square (0 = 0^2), any product with 0 is 0 (a perfect square).
    # So the number of such pairs is:
    #   (pairs where one is zero and the other is nonzero) = zero_count * (n - zero_count)
    # plus (pairs where both are zeros) = (zero_count choose 2)
    total_pairs = zero_count * (n - zero_count) + (zero_count * (zero_count - 1)) // 2

    # For nonzero values, we can show that:
    #   A_i * A_j is a perfect square if and only if the squarefree part of A_i equals that of A_j.
    # Write A as: A = (square part)^2 * (squarefree part).
    # Then, A_i * A_j = (square parts product)^2 * (squarefree_i * squarefree_j).
    # Since squarefree_i and squarefree_j are squarefree, their product is a square only when they are equal.
    #
    # Thus, for nonzero values, we group them by their squarefree representation.
    # Then for any group with frequency count f, we have (f choose 2) valid pairs.
    
    if nonzero:
        max_val = max(nonzero)
        # Precompute smallest prime factor (spf) for numbers up to max_val.
        spf = list(range(max_val + 1))
        r = int(math.isqrt(max_val)) + 1
        for i in range(2, r):
            if spf[i] == i:  # i is prime
                for j in range(i * i, max_val + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        def get_squarefree(x):
            # Compute the squarefree part of x (x > 0).
            res = 1
            while x > 1:
                p = spf[x]
                cnt = 0
                while x % p == 0:
                    cnt += 1
                    x //= p
                if cnt % 2 == 1:
                    res *= p
            return res

        freq = {}
        for a in nonzero:
            sf = get_squarefree(a)
            freq[sf] = freq.get(sf, 0) + 1
            
        for count in freq.values():
            total_pairs += count * (count - 1) // 2

    sys.stdout.write(str(total_pairs))
    
if __name__ == '__main__':
    main()