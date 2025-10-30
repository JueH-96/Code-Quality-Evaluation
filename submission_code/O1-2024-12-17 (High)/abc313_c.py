def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Sum of all A_i
    S = sum(A)

    # M1 = floor(S/N)
    M1 = S // N
    # r = how many elements must be exactly M1+1 instead of M1
    r = S - N * M1  # same as S % N

    # Compute the total "surplus" above M1 and how many elements exceed M1
    # surplus T_s = sum(max(0, A_i - M1))
    # c = count of A_i such that A_i > M1
    T_s = 0
    c = 0
    for x in A:
        if x > M1:
            T_s += x - M1
            c += 1

    # If r=0, then all elements must be M1, so the cost is just T_s.
    # Otherwise, we can reduce the surplus by min(r,c) because each of those r
    # increments (to M1+1) can come from an element above M1, saving 1 operation each time.
    if r == 0:
        cost = T_s
    else:
        cost = T_s - min(r, c)

    print(cost)

# Do not forget to call main()
if __name__ == "__main__":
    main()