def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    L = [0]*N
    R = [0]*N

    # Parse input
    idx = 1
    for i in range(N):
        L[i] = int(data[idx]); R[i] = int(data[idx+1])
        idx += 2

    # Compute sum of all L_i and sum of all R_i
    sumL = sum(L)
    sumR = sum(R)

    # Check feasibility:
    # If sumL > 0, we can't reduce any X_i below L_i, so we can't get sum=0.
    # If sumR < 0, even maximally picking R_i won't reach sum=0.
    if sumL > 0 or sumR < 0:
        print("No")
        return

    # We know we can form a valid sequence. We'll start from X_i = L_i for all i
    # and distribute the necessary increments to reach sum=0.
    X = [0]*N
    for i in range(N):
        X[i] = L[i]

    current_sum = sumL
    # If current_sum < 0, we need to add (-current_sum) across X's as possible.
    needed = -current_sum  # This is how much we must add to get sum=0

    for i in range(N):
        if needed <= 0:
            break
        can_add = R[i] - L[i]
        add = min(can_add, needed)
        X[i] += add
        needed -= add

    # At this point, the sum of X should be 0.
    print("Yes")
    print(" ".join(map(str, X)))


# Do not forget to call main() at the end
if __name__ == "__main__":
    main()