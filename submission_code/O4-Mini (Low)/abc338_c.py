import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Q = [int(next(it)) for _ in range(N)]
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    
    INF = 10**18
    
    # f(A, B) will iterate over x (servings of dish A),
    # compute max y (servings of dish B) for each x,
    # and return the maximum x+y.
    def f(A, B):
        # compute the maximum possible x by ingredients A
        max_x = INF
        for i in range(N):
            if A[i] > 0:
                # we can make at most Q[i]//A[i] servings of A from ingredient i
                max_x = min(max_x, Q[i] // A[i])
        # max_x is now the upper bound for x
        best = 0
        # iterate x = 0..max_x
        for x in range(int(max_x) + 1):
            y_max = INF
            ok = True
            # for each ingredient, check remainder after x servings of A
            for i in range(N):
                rem = Q[i] - A[i] * x
                if rem < 0:
                    ok = False
                    break
                # if B[i]>0, it limits y
                if B[i] > 0:
                    # we can make at most rem//B[i] servings of B
                    y_max = min(y_max, rem // B[i])
            if not ok:
                # as x grows, rem only decreases, so we could break entirely here,
                # but since different A[i] might be zero we can't guarantee early break
                continue
            # if B has some zero entries, y_max may still be INF, but we know
            # there is at least one B[i]>0 so y_max is finite
            best = max(best, x + (y_max if y_max < INF else 0))
        return best

    # Because x+y is symmetric, try iterating over x (A first) or y (B first)
    ans = max(f(A, B), f(B, A))
    print(ans)

if __name__ == "__main__":
    main()