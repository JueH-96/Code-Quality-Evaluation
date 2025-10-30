def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Keep inserting until all adjacent diffs are exactly 1
    while True:
        n = len(A)
        idx = -1
        # find first adjacent pair with |diff| != 1
        for i in range(n - 1):
            if abs(A[i] - A[i+1]) != 1:
                idx = i
                break
        if idx == -1:
            break  # all differences are 1, we're done

        # build new sequence with the required insertions at position idx
        a, b = A[idx], A[idx+1]
        new_seq = []
        # copy up through A[idx]
        new_seq.extend(A[:idx+1])

        # insert the intermediate numbers
        if a < b:
            # ascend from a+1 up to b-1
            new_seq.extend(range(a+1, b))
        else:
            # descend from a-1 down to b+1
            new_seq.extend(range(a-1, b, -1))

        # append the remainder from A[idx+1] onward
        new_seq.extend(A[idx+1:])
        A = new_seq

    # output the final sequence
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()