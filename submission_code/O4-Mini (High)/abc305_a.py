def main():
    N = int(input().strip())
    # There are water stations at every 5 km: 0, 5, 10, ..., 100.
    # Find the nearest multiple of 5 to N.
    q, r = divmod(N, 5)
    if r < 3:
        print(q * 5)
    else:
        print(q * 5 + 5)

if __name__ == "__main__":
    main()