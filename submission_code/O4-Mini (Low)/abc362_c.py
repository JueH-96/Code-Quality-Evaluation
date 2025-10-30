import sys
import threading

def main():
    import sys
    data = sys.stdin
    N = int(data.readline().strip())
    L = [0]*N
    R = [0]*N
    sum_L = 0
    sum_R = 0
    for i in range(N):
        li, ri = map(int, data.readline().split())
        L[i] = li
        R[i] = ri
        sum_L += li
        sum_R += ri

    # We need sum X_i = 0, with L_i <= X_i <= R_i.
    # The possible total sum is in [sum_L, sum_R].
    if not (sum_L <= 0 <= sum_R):
        print("No")
        return

    # We'll start from X_i = L_i, so current sum = sum_L.
    # We need to increase by rem = -sum_L to reach 0.
    rem = -sum_L
    X = [0]*N
    for i in range(N):
        # Maximum we can add on top of L[i].
        avail = R[i] - L[i]
        add = avail if avail < rem else rem
        X[i] = L[i] + add
        rem -= add
        # Once rem hits 0 we can copy the rest as L
        if rem == 0:
            for j in range(i+1, N):
                X[j] = L[j]
            break

    # rem must be zero now
    print("Yes")
    print(" ".join(map(str, X)))

if __name__ == "__main__":
    main()