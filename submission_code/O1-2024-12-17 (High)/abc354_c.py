def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    cards = []
    idx = 1
    for i in range(N):
        A = int(data[idx])
        C = int(data[idx+1])
        idx += 2
        cards.append((A, C, i+1))

    # Sort cards by descending strength A
    cards.sort(key=lambda x: x[0], reverse=True)

    # We'll track the minimum cost seen so far among cards with strictly greater A
    min_cost = float('inf')
    kept_indices = []

    # If a card's cost is lower than all previously encountered costs (from higher A),
    # then it is not dominated and should remain.
    # Otherwise, it is discarded.
    for A, C, i in cards:
        if C < min_cost:
            kept_indices.append(i)
            min_cost = C

    # Sort the kept cards by their original index and output
    kept_indices.sort()
    print(len(kept_indices))
    print(" ".join(map(str, kept_indices)))

# Do not forget to call main()
if __name__ == "__main__":
    main()