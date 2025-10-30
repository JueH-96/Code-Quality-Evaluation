def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data: 
        return
    N = int(data[0])
    
    # A positive integer X is called "good" if there exists a pair of positive integers (a, b)
    # such that X = 2^a * b^2.
    # Notice that a must be at least 1 so every good integer is even.
    # Write any even number X in its 2-adic form: X = 2^d * y, where y is odd.
    # Then X = 2^a * (b')^2 with b' odd if and only if we have:
    #     2^d * y = 2^(a + 2k) * m^2,
    # where b = 2^k * m and m is odd.
    # So we must have:
    #   d = a + 2k   with a >=1, k >=0, and y = m^2 (i.e. y must be a perfect square).
    # Thus, a number X is good if and only if it is even and its odd part (when factoring out all 2’s)
    # is a perfect square.
    #
    # We can count these numbers by noticing that any good integer X can be uniquely represented as:
    #    X = 2^d * m^2,
    # where d >= 1 and m is a positive odd integer.
    # For X <= N, we have:
    #    2^d * m^2 <= N  ==>  m^2 <= N / 2^d.
    #
    # For any fixed d >= 1, let L = N // (2^d) and m_max = floor(sqrt(L)).
    # We need to count only the odd m among 1..m_max, and the number of odd numbers in that range is:
    #    count = (m_max + 1) // 2.
    #
    # We iterate d from 1 until N // (2^d) becomes 0.
    
    ans = 0
    d = 1
    while True:
        L = N // (2 ** d)
        if L <= 0:
            break
        m_max = math.isqrt(L)
        # Count of odd numbers <= m_max is (m_max + 1) // 2
        ans += (m_max + 1) // 2
        d += 1
    
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()