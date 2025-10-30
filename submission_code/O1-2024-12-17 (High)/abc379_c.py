def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    X = list(map(int, data[2:2+M]))
    A = list(map(int, data[2+M:2+2*M]))

    # If total stones is not exactly N, it's impossible
    if sum(A) != N:
        print(-1)
        return

    # Pair up the positions with the corresponding stone counts and sort by position
    stones = sorted(zip(X, A))

    cost = 0
    # i represents the next cell in the final arrangement (1 to N) that we need to fill
    i = 1

    for x, a in stones:
        # If there's a gap we cannot fill (x > i means we missed filling cell i)
        if i < x:
            print(-1)
            return

        # If there are still cells to fill, use as many stones from this group as needed
        if i <= N:
            # How many cells can we fill from this group?
            fill_count = min(a, N - i + 1)

            # Each of these fill_count stones moves from position x to positions i, i+1, ..., i+fill_count-1
            # The total distance traveled by these stones is:
            #   sum_{k=0..(fill_count-1)} ( (i + k) - x )
            # = fill_count*(i - x) + sum_{k=0..fill_count-1} k
            # = fill_count*(i - x) + (fill_count*(fill_count - 1))//2
            cost += fill_count * (i - x)
            cost += (fill_count * (fill_count - 1)) // 2

            i += fill_count

        # If we've filled all cells, we can stop
        if i > N:
            break

    # If we haven't filled up to cell N, it's impossible
    if i <= N:
        print(-1)
    else:
        print(cost)

# Do not forget to call main() at the end
main()