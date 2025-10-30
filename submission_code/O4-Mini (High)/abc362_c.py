import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N_line = input().strip()
    if not N_line:
        print("No")
        return
    N = int(N_line)
    L = [0] * N
    R = [0] * N
    sum_min = 0           # sum of L_i
    sum_max = 0           # sum of R_i
    for i in range(N):
        li, ri = map(int, input().split())
        L[i] = li
        R[i] = ri
        sum_min += li
        sum_max += ri

    # We need sum X_i = 0, and each X_i in [L_i, R_i].
    # A necessary and sufficient condition is sum_min <= 0 <= sum_max.
    if not (sum_min <= 0 <= sum_max):
        print("No")
        return

    # Start with X_i = L_i (sum = sum_min), then raise some X_i up to R_i
    # to make the total sum zero.
    D = -sum_min   # the total increment we need to add to reach sum = 0
    X = [0] * N
    for i in range(N):
        cap = R[i] - L[i]
        inc = cap if cap < D else D
        X[i] = L[i] + inc
        D -= inc
        # once D reaches 0, the rest X[i]=L[i] (which is already ensured if inc=0)
    # At this point D must be 0 because sum_min <= 0 <= sum_max implies
    # total capacity sum >= -sum_min = D

    print("Yes")
    # print the sequence
    out = " ".join(str(x) for x in X)
    print(out)

if __name__ == "__main__":
    main()