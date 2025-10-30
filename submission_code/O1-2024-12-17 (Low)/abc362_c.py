def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    LR = input_data[1:]
    
    L = []
    R = []
    idx = 0
    for _ in range(N):
        l_i = int(LR[idx])
        r_i = int(LR[idx+1])
        L.append(l_i)
        R.append(r_i)
        idx += 2

    sumL = sum(L)
    sumR = sum(R)

    # If 0 is not in the range [sumL, sumR], no solution
    if sumL > 0 or sumR < 0:
        print("No")
        return

    # Start with X_i = L_i
    X = L[:]
    current_sum = sumL

    # We want to make current_sum = 0 by distributing increments
    to_fix = -current_sum  # how much we need to add to reach 0
    for i in range(N):
        if to_fix <= 0:
            break
        can_add = R[i] - L[i]
        add_now = min(can_add, to_fix)
        X[i] += add_now
        to_fix -= add_now

    # After distribution, if the sum is not 0, then no solution
    if to_fix != 0:
        print("No")
        return

    # Otherwise, we have a valid sequence
    print("Yes")
    print(" ".join(map(str, X)))


# Call main() at the end
main()