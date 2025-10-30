def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    mod = 998244353
    N = int(data[0])
    A = list(map(int, data[1:]))

    # It turns out that the conditions force that for each i,
    # the number P_i must be the minimum in the segment [A_i+1, i]
    # (with the understanding that if A_i=0 then there is no left neighbor condition).
    # A short analysis shows that the number of valid permutations is given by
    #   ∏ᵢ₌₁ᴺ (i - A_i)
    # (see explanation at the end).
    #
    # To check the sample: for N=4, A=[0,1,0,3] we get:
    #   (1-0) * (2-1) * (3-0) * (4-3) = 1 * 1 * 3 * 1 = 3,
    # which is the sample output.
    #
    # Hence, we compute the product modulo 998244353.

    ans = 1
    for i in range(1, N+1):
        # A is 0-indexed, and A[i-1] satisfies 0 <= A[i-1] < i.
        ans = (ans * (i - A[i-1])) % mod

    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()