def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # A good integer X is one for which there exist positive integers a and b with
    # X = 2^a * b^2.
    # It turns out that every such X can be written uniquely in the form:
    #      X = 2^r * d^2   (with r >= 1 and d odd)
    # because if b has any factors of 2 we can absorb them into the 2^a part.
    #
    # Thus, an integer X is good (X ≥ 1) if and only if it is even and, writing X = 2^r * m
    # where m is odd, m is a perfect square.
    #
    # Equivalently, the good integers are exactly those of the form:
    #      X = 2^a * k^2   with a >= 1 and k an odd positive integer, and X <= N.
    #
    # So we need to count all pairs (a, k) such that:
    #      2^a * k^2 <= N,  a >= 1, and k odd.
    #
    # For a fixed a (with 2^a <= N), the condition on k is:
    #      k^2 <= N / (2^a)
    # Meaning that k must be at most floor( sqrt( N/(2^a) ) ).
    # Also, we count only the odd k in the range [1, floor(sqrt(N/2^a))].
    #
    # The number of odd integers from 1 to t (inclusive) is (t + 1) // 2. 
    # (For example, if t = 7 then the odds are 1,3,5,7: count = (7+1)//2 = 4.)
    
    # We loop over a = 1,2,... while 2^a <= N, and for each a we add:
    #      count = (floor(sqrt(N//(2^a))) + 1) // 2
    ans = 0
    a = 1
    while True:
        power = 1 << a  # 2^a
        if power > N:
            break
        # For fixed a, let limit = floor( N / (2^a) )
        limit = N // power
        if limit <= 0:
            break
        # Using math.isqrt to get the floor of the square root of limit.
        t = math.isqrt(limit)
        if t <= 0:
            break
        # Count the odd numbers in 1..t.
        count_odds = (t + 1) // 2
        ans += count_odds
        a += 1

    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()