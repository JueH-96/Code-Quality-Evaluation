def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    cards = []
    index = 1
    for i in range(N):
        A = int(data[index])
        C = int(data[index + 1])
        cards.append((A, C, i + 1))
        index += 2
    
    # Sort cards by A in descending order
    cards_sorted = sorted(cards, key=lambda x: (-x[0]))
    
    min_C = float('inf')
    remaining = []
    for card in cards_sorted:
        if card[1] < min_C:
            remaining.append(card[2])
            min_C = card[1]
    
    # Sort the remaining indices in ascending order
    remaining_sorted = sorted(remaining)
    print(len(remaining_sorted))
    print(' '.join(map(str, remaining_sorted)))

if __name__ == '__main__':
    main()