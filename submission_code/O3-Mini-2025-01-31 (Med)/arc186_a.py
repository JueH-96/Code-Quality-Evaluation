def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    Q = int(data[1])
    tot = N * N
    # For a 0–1 matrix with prescribed row–sums and column–sums,
    # one may prove that if there is more than one solution then every entry may be switched.
    # In other words, a matrix with non–extreme margins has no fixed entry.
    # On the other hand, if at least one row or column has an “extreme” margin (that is, 0 or N)
    # then consistency forces every row and every column to be extreme and the table is unique.
    # In that case every cell is forced to its unique value.
    # Hence the only two possible numbers of fixed entries are 0 and N^2.
    #
    # (A detailed explanation of this “all–or–nothing” phenomenon may be found in
    # the literature on 0–1 contingency tables.)
    
    # So, we answer "Yes" exactly when K is 0 or K equals N^2,
    # and "No" otherwise.
    #
    # Read the Q queries and output the answer for each.
    out_lines = []
    for idx in range(2, 2+Q):
        K = int(data[idx])
        if K == 0 or K == tot:
            out_lines.append("Yes")
        else:
            out_lines.append("No")
    sys.stdout.write("
".join(out_lines))
    

if __name__ == '__main__':
    main()