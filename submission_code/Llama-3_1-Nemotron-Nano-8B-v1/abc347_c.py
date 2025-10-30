def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = int(input[1])
    B = int(input[2])
    D = list(map(int, input[3:3+N]))
    M = A + B
    max_L = 0
    min_R = M - 1
    intervals = []
    for d in D:
        s = d % M
        L = (A - 1 - s) % M
        R = (L + A - 1) % M
        intervals.append((L, R))
        if L > max_L:
            max_L = L
        if R < min_R:
            min_R = R
    if max_L <= min_R:
        print("Yes")
        return
    valid = True
    for L, R in intervals:
        if not (R >= max_L or (L <= max_L and R >= M)):
            valid = False
            break
    if valid:
        print("Yes")
        return
    valid = True
    for L, R in intervals:
        if not (L <= min_R or (R >= 0 and L <= min_R)):
            valid = False
            break
    if valid:
        print("Yes")
        return
    print("No")

if __name__ == "__main__":
    main()