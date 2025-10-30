def main() -> None:
    import sys

    N = int(sys.stdin.readline().strip())

    repunits = []            # repunits[k]  ->  the number 11..1 with (k+1) digits
    current = 0              # last generated repunit
    length = 0               # how many different lengths we already have

    while True:
        # add the next repunit (with one more digit 1)
        length += 1
        current = current * 10 + 1          # 1, 11, 111, ...
        repunits.append(current)

        # build every possible sum of exactly three (not-necessarily-distinct) repunits
        sums = set()
        for i in range(length):
            for j in range(i, length):
                for k in range(j, length):
                    sums.add(repunits[i] + repunits[j] + repunits[k])

        sums_sorted = sorted(sums)

        # if we already have at least N distinct sums,
        # check whether any still-missing sums (i.e. sums that use a longer repunit)
        # can be smaller than the current N-th element
        if len(sums_sorted) >= N:
            next_repunit = current * 10 + 1      # the first repunit with (length+1) digits
            smallest_future_sum = next_repunit + 2  # 1 + 1 + next_repunit
            if sums_sorted[N - 1] < smallest_future_sum:
                print(sums_sorted[N - 1])
                return


if __name__ == "__main__":
    main()