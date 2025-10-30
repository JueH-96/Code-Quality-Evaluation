def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:]))

    # The goal is to find the minimum x such that P[0] + x > max(P[1:])
    # That is x = max(0, (max(P[1:]) - P[0] + 1)).
    # If person 1 is already the strongest (or tied for strongest), no extra points are needed.

    if N == 1:
        # If there's only one person, they are trivially the strongest.
        print(0)
        return

    max_others = max(P[1:])
    x = max(0, max_others - P[0] + 1)
    print(x)

def main():
    solve()

if __name__ == "__main__":
    main()