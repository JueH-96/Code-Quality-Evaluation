def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    Ds = list(map(int, data[1:]))

    total = 0
    for i in range(1, N+1):
        # Check if month i is a repdigit: all digits same
        si = str(i)
        if all(ch == si[0] for ch in si):
            d = int(si[0])
            # Generate repdigit days j of the same digit d: d, dd, ddd, ...
            j = d
            idx = i - 1  # index into Ds
            while j <= Ds[idx]:
                total += 1
                j = j * 10 + d
    print(total)

if __name__ == "__main__":
    main()