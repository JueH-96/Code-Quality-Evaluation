import sys
import threading

def main():
    import sys
    data = sys.stdin
    n_line = data.readline().strip()
    if not n_line:
        print("No")
        return
    N = int(n_line)
    L = [0]*N
    R = [0]*N
    sumL = 0
    sumR = 0
    for i in range(N):
        li, ri = map(int, data.readline().split())
        L[i] = li
        R[i] = ri
        sumL += li
        sumR += ri
    # We need sum_i X_i = 0 with L_i <= X_i <= R_i.
    # The possible total sum lies in [sumL, sumR].
    if sumL > 0 or sumR < 0:
        print("No")
        return
    # Start with X_i = L_i, current sum = sumL. We need to add D = -sumL to reach 0.
    D = -sumL
    X = [0]*N
    for i in range(N):
        # We can increase this variable by at most (R[i] - L[i])
        inc = R[i] - L[i]
        if inc >= D:
            # Use only what's needed
            X[i] = L[i] + D
            D = 0
        else:
            X[i] = R[i]
            D -= inc
    # At this point, D must be 0.
    # Print solution
    print("Yes")
    # join the integers with spaces
    out = " ".join(str(x) for x in X)
    print(out)

if __name__ == "__main__":
    main()