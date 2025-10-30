def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # c0(j) = number of subintervals ending at j whose NAND-fold is 0
    # c1(j) = number of subintervals ending at j whose NAND-fold is 1
    #
    # For j ≥ 1:
    #   if S[j-1] == '0':
    #       c0(j) = 1             # the new subinterval j..j is "0"
    #       c1(j) = j-1          # all subintervals from previous step become 1
    #   else (S[j-1] == '1'):
    #       c0(j) = c1(j-1)      # those that were 1 become 0
    #       c1(j) = c0(j-1) + 1  # those that were 0 become 1, plus new subinterval (1)
    #
    # The answer is the sum of c1(j) for j = 1..N, because f(i,j) ∈ {0,1} and
    # we want the sum of all f(i,j); that is the count of subintervals
    # whose fold is 1.

    answer = 0
    c0, c1 = 0, 0  # c0(0) and c1(0) both 0 (no subintervals at j=0)

    for j in range(1, N + 1):
        if S[j - 1] == '0':
            c0_new = 1
            c1_new = j - 1
        else:  # S[j-1] == '1'
            c0_new = c1
            c1_new = c0 + 1

        c0, c1 = c0_new, c1_new
        answer += c1

    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()