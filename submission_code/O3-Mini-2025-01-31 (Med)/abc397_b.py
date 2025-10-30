def main():
    import sys
    S = sys.stdin.read().strip()
    n = len(S)

    # We are allowed to insert characters into S to form a string T that
    # satisfies:
    #   1. |T| is even.
    #   2. The odd-numbered characters (positions 1, 3, …) are "i" and even-numbered characters (2, 4, …) are "o".
    # Note that T is completely determined by its length L: T = "i", "o", "i", "o", ..., up to L.
    # Since insertions can only increase the length, we are looking for an even L >= n such that S is a subsequence of T.
    # The minimum number of insertions is then L - n.
    #
    # We can bound L by 2*n. For instance, in the worst-case if S consists only of "i"
    # then T must have enough "i" positions. Since T of even length L will have L/2 "i"s,
    # we need L/2 >= (count of "i" in S). In the worst-case count("i") = n, so L must be at most 2*n.
    #
    # We iterate for even L from the smallest even number >= n to 2*n, and check if S is
    # a subsequence of T (which is just the alternating pattern "i", "o", "i", "o", ...).
    
    # Get the minimum even number not less than n.
    L_start = n if n % 2 == 0 else n + 1

    # Iterate over candidate T lengths
    result = None
    for L in range(L_start, 2 * n + 1, 2):
        idx = 0  # pointer for S
        for j in range(L):
            # Character at position j in T (0-indexed): even positions => "i", odd => "o"
            expected = 'i' if j % 2 == 0 else 'o'
            if idx < n and S[idx] == expected:
                idx += 1
            if idx == n:
                break
        if idx == n:  # S is a subsequence of T of length L
            result = L - n
            break

    # Under the problem guarantees, result will always be set.
    print(result)

if __name__ == '__main__':
    main()