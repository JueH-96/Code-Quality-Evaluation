def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))

    # right[v] = how many times v appears to the right of current position
    # left[v]  = how many times v appears to the left of current position
    right = [0] * (N + 1)
    for v in A:
        right[v] += 1
    left = [0] * (N + 1)

    # sum_all = sum over all values x of (left[x] * right[x])
    sum_all = 0
    ans = 0

    for v in A:
        L = left[v]
        R = right[v]

        # At this position j with value v,
        # the number of valid triples with middle index j is
        #   sum_all - (L * R)
        # because we exclude the term where the middle value equals the endpoints' value.
        ans += sum_all - L * R

        # Now we move this A[j]=v from "right" to "left",
        # so we update sum_all by the change in the product left[v]*right[v].
        # old product = L * R
        # new product = (L+1)*(R-1) = L*R + R - L - 1
        # delta = (R - L - 1)
        sum_all += (R - L - 1)

        # update counts
        left[v] += 1
        right[v] -= 1

    print(ans)

if __name__ == "__main__":
    main()