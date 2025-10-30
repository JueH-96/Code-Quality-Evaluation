def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Coordinate compression
    unique_vals = sorted(set(A))
    rank = {v: i+1 for i, v in enumerate(unique_vals)}

    # Fenwicks (Binary Indexed Trees) for counts and sums
    size = len(unique_vals)
    fenw_count = [0]*(size+1)
    fenw_sum = [0]*(size+1)

    def fenw_update(fenw, i, val):
        while i <= size:
            fenw[i] += val
            i += i & -i

    def fenw_query(fenw, i):
        s = 0
        while i > 0:
            s += fenw[i]
            i -= i & -i
        return s

    answer = 0
    for x in A:
        r = rank[x]
        # Query how many previous values are strictly less than x
        c = fenw_query(fenw_count, r - 1)  # count
        s = fenw_query(fenw_sum, r - 1)    # sum
        answer += c*x - s
        # Update Fenwicks
        fenw_update(fenw_count, r, 1)
        fenw_update(fenw_sum, r, x)

    print(answer)

# Do not forget to call main.
if __name__ == "__main__":
    main()