def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    tot = [0] * 24
    idx = 1
    for _ in range(N):
        w = int(data[idx]); x = int(data[idx+1])
        idx += 2
        # We need (t + x) % 24 in [9,17]
        # => t % 24 in [9 - x, 17 - x] mod 24
        s = (9 - x) % 24
        e = (17 - x) % 24
        if s <= e:
            # non-wrapping interval
            for t in range(s, e+1):
                tot[t] += w
        else:
            # wrapping around midnight
            for t in range(s, 24):
                tot[t] += w
            for t in range(0, e+1):
                tot[t] += w

    # The answer is the maximum number of employees over all possible start times
    print(max(tot))


if __name__ == "__main__":
    main()