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
    # Compute maximum possible x (number of A servings) and y (number of B servings)
    INF = 10**18
    max_x = INF
    for i in range(N):
        if A[i] > 0:
            # Can make at most Q[i]//A[i]
            v = Q[i] // A[i]
            if v < max_x:
                max_x = v
        # if A[i]==0, no bound from this ingredient
    max_y = INF
    for i in range(N):
        if B[i] > 0:
            v = Q[i] // B[i]
            if v < max_y:
                max_y = v
    # They guarantee there's at least one positive A[i] and one positive B[i], so both finite
    ans = 0
    # Decide which to iterate
    if max_x < max_y:
        # iterate x in [0..max_x]
        for x in range(max_x + 1):
            # check feasibility and compute best y
            y_max = max_y
            ok = True
            # For each ingredient, enforce A[i]*x + B[i]*y <= Q[i]
            for i in range(N):
                needA = A[i] * x
                if needA > Q[i]:
                    ok = False
                    break
                if B[i] > 0:
                    # y <= (Q[i] - needA) // B[i]
                    v = (Q[i] - needA) // B[i]
                    if v < y_max:
                        y_max = v
                        # early stop if y_max < 0
                        if y_max < 0:
                            ok = False
                            break
                # if B[i]==0, no bound on y from i
            if not ok:
                continue
            # y_max is >= 0
            total = x + y_max
            if total > ans:
                ans = total
    else:
        # iterate y in [0..max_y]
        for y in range(max_y + 1):
            x_max_here = max_x
            ok = True
            for i in range(N):
                needB = B[i] * y
                if needB > Q[i]:
                    ok = False
                    break
                if A[i] > 0:
                    v = (Q[i] - needB) // A[i]
                    if v < x_max_here:
                        x_max_here = v
                        if x_max_here < 0:
                            ok = False
                            break
                # if A[i]==0, no bound on x from i
            if not ok:
                continue
            total = y + x_max_here
            if total > ans:
                ans = total
    # Print result
    print(ans)

if __name__ == "__main__":
    main()