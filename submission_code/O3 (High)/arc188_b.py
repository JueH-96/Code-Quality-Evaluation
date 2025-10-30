import sys

def main() -> None:
    rd = sys.stdin.readline
    T = int(rd())
    ans = []
    for _ in range(T):
        N, K = map(int, rd().split())

        # When the two players stand at diametrically opposite points
        # (this happens only if N is even and K = N/2), both of their
        # symmetry axes are identical.  After the two points on this
        # common axis have been painted black, every remaining white
        # point is paired with another white point and neither player
        # can paint only one of them without violating the symmetry
        # requirement.  Hence no further move is possible and not all
        # points become black.  (The only exception is N = 2, where
        # there are no off-axis points at all and both points can be
        # painted.)
        #
        # In every other situation the two axes are different.  The
        # players can then alternately make the mirror image of the
        # last coloured point with respect to their own axis black.
        # This procedure always allows another legal move until every
        # point is coloured.
        #
        # Consequently the answer is NO iff
        #     – N is even,
        #     – K = N/2,
        #     – and N > 2.
        # In every other case it is YES.
        if N == 2:
            ans.append('Yes')
        elif N % 2 == 0 and K == N // 2:
            ans.append('No')
        else:
            ans.append('Yes')

    print('
'.join(ans))

if __name__ == "__main__":
    main()