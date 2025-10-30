def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:]))

    # Find the maximum ability among persons 2..N (if any)
    if n > 1:
        max_other = max(P[1:])
    else:
        max_other = 0

    # We need P[0] + x > max_other  =>  x > max_other - P[0]
    # Minimum non-negative x is max(0, max_other - P[0] + 1)
    ans = max(0, max_other - P[0] + 1)
    print(ans)

if __name__ == "__main__":
    main()