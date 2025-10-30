import sys

def main():
    N = int(sys.stdin.readline())
    cards = []
    for idx in range(1, N+1):
        A, C = map(int, sys.stdin.readline().split())
        cards.append((A, C, idx))
    
    # Sort the cards by strength (A) in increasing order
    sorted_cards = sorted(cards, key=lambda x: x[0])
    n = len(sorted_cards)
    
    # Compute suffix_min array
    suffix_min = [0] * n
    suffix_min[-1] = float('inf')
    for i in range(n-2, -1, -1):
        next_C = sorted_cards[i+1][1]
        suffix_min[i] = min(next_C, suffix_min[i+1])
    
    # Determine which cards are kept
    kept = []
    for i in range(n):
        a, c, idx = sorted_cards[i]
        if suffix_min[i] >= c:
            kept.append(idx)
    
    # Sort the kept indices and output
    kept.sort()
    print(len(kept))
    print(' '.join(map(str, kept)))

if __name__ == "__main__":
    main()