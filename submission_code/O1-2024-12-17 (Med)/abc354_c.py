def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    cards = []
    for i in range(N):
        A, C = map(int, input().split())
        cards.append((A, C, i + 1))

    # Sort by strength (A) in ascending order
    cards.sort(key=lambda x: x[0])

    # We'll move from the largest strength to the smallest,
    # keeping track of the smallest cost seen so far.
    min_cost_so_far = float('inf')
    survivors = []

    # Traverse in descending order of A
    for A, C, idx in reversed(cards):
        if C < min_cost_so_far:
            survivors.append((idx, A, C))
            min_cost_so_far = C

    # Sort survivors by their original index
    survivors.sort(key=lambda x: x[0])

    print(len(survivors))
    print(" ".join(str(s[0]) for s in survivors))

# Do NOT forget to call main()
if __name__ == "__main__":
    main()