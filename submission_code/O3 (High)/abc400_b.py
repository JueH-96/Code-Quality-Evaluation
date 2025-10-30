import sys

def main() -> None:
    LIMIT = 10 ** 9          # 1,000,000,000
    
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N, M = map(int, data)

    total = 0
    term = 1                 # N^0

    for i in range(M + 1):
        # Will adding this term exceed the limit?
        if total + term > LIMIT:
            print("inf")
            return
        total += term

        # No more terms to generate
        if i == M:
            break

        # If the next term is guaranteed to exceed LIMIT, we can stop early
        if term > LIMIT // N:
            print("inf")
            return

        term *= N            # Prepare N^(i+1)

    print(total)


if __name__ == "__main__":
    main()