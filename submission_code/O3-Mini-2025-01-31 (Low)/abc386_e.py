def main():
    import sys
    from itertools import combinations

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    arr = [int(next(it)) for _ in range(n)]

    max_xor = 0
    # Since binom(n, k) <= 10^6 we can iterate over all combinations.
    for combo in combinations(arr, k):
        x = 0
        for num in combo:
            x ^= num
        if x > max_xor:
            max_xor = x

    sys.stdout.write(str(max_xor))


if __name__ == "__main__":
    main()