def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    pairs = []
    sumL = 0
    sumR = 0
    index = 1
    for i in range(N):
        L = int(data[index])
        R = int(data[index+1])
        index += 2
        pairs.append((L, R))
        sumL += L
        sumR += R

    # Check if 0 can be expressed as a sum of X_i with X_i in [L_i, R_i].
    if sumL > 0 or sumR < 0:
        sys.stdout.write("No")
        return

    # Start by taking X_i = L_i for all i.
    # Then we need to add exactly D = -sumL to reach a total sum of zero.
    D = -sumL
    result = []
    for L, R in pairs:
        capacity = R - L  # Maximum amount we can add for this element.
        add = min(D, capacity)
        result.append(L + add)
        D -= add

    # At this point, D must be zero if a valid sequence is found.
    if D != 0:
        sys.stdout.write("No")
    else:
        sys.stdout.write("Yes
" + " ".join(map(str, result)))
        
if __name__ == '__main__':
    main()