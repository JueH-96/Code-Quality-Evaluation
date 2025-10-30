def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    
    L = []
    R = []
    idx = 1
    for _ in range(N):
        l_i = int(data[idx])
        r_i = int(data[idx + 1])
        L.append(l_i)
        R.append(r_i)
        idx += 2
    
    sumL = sum(L)
    sumR = sum(R)
    
    # If 0 isn't in the range [sumL, sumR], no solution exists.
    if sumL > 0 or sumR < 0:
        print("No")
        return
    
    # We'll start with X_i = L_i for each i.
    X = L[:]
    # The amount we need to increase the total sum from sumL to 0.
    needed = -sumL  # This will be >= 0 if 0 is in [sumL, sumR].
    
    # Greedily add increments to match the needed sum.
    for i in range(N):
        if needed == 0:
            break
        can_add = R[i] - L[i]  # Max possible increment at position i
        actual_add = min(can_add, needed)
        X[i] += actual_add
        needed -= actual_add
    
    # If after all increments we haven't achieved 0 total, no solution.
    if needed != 0:
        print("No")
    else:
        print("Yes")
        print(" ".join(map(str, X)))

# Do not forget to call main()
if __name__ == "__main__":
    main()