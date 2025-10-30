import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    cards = []
    for i in range(N):
        A = int(data[1 + 2*i])
        C = int(data[2 + 2*i])
        index = i + 1  # 1-based indexing
        cards.append((A, C, index))
    
    # Sort cards in decreasing order of A_i
    sorted_cards = sorted(cards, key=lambda x: -x[0])
    
    min_C = float('inf')
    kept = []
    
    for card in sorted_cards:
        if card[1] < min_C:
            kept.append(card[2])
            min_C = card[1]
    
    # Sort the kept indices in ascending order
    kept_sorted = sorted(kept)
    
    # Print the results
    print(len(kept_sorted))
    print(' '.join(map(str, kept_sorted)))

if __name__ == "__main__":
    main()