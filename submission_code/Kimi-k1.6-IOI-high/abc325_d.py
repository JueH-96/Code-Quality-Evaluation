def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    products = []
    idx = 1
    for _ in range(n):
        T = int(data[idx])
        D = int(data[idx+1])
        end = T + D
        products.append((end, T))
        idx += 2
    # Sort by end time, then by T
    products.sort()
    count = 0
    last = -1  # Initialize to a value lower than any possible T (which is >=1)
    for end, T in products:
        earliest = max(T, last + 1)
        if earliest <= end:
            count += 1
            last = earliest
    print(count)

if __name__ == "__main__":
    main()