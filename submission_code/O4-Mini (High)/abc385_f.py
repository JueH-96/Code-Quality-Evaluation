#!/usr/bin/env python3
import sys
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    X = [0]*n
    H = [0]*n
    for i in range(n):
        X[i] = int(next(it))
        H[i] = int(next(it))
    # We need the maximum intercept
    # For adjacent buildings i-1,i the intercept c is
    #   c = (H[i-1]*X[i] - H[i]*X[i-1]) / (X[i] - X[i-1])
    # It can be shown the global maximum over all pairs j<i
    # equals the maximum over adjacent pairs.
    best_num = None
    best_den = 1
    for i in range(1, n):
        num = H[i-1]*X[i] - H[i]*X[i-1]
        den = X[i] - X[i-1]  # > 0 since X strictly increasing
        # compare num/den with best_num/best_den
        if best_num is None or num * best_den > best_num * den:
            best_num = num
            best_den = den
    # If the maximum intercept is negative, even height 0 suffices => -1
    if best_num is None or best_num < 0:
        sys.stdout.write("-1")
    else:
        ans = best_num / best_den
        # print with sufficient precision (error <=1e-9)
        sys.stdout.write("{:.12f}".format(ans))

if __name__ == "__main__":
    main()