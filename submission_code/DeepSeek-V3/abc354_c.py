def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = []
    C = []
    index = 1
    for i in range(N):
        A.append(int(data[index]))
        C.append(int(data[index+1]))
        index += 2
    
    # Pair each card with its original index (1-based)
    cards = list(zip(A, C, range(1, N+1)))
    
    # Sort cards by strength in ascending order
    cards.sort()
    
    remaining = []
    min_cost = float('inf')
    
    for card in cards:
        a, c, idx = card
        if c < min_cost:
            remaining.append(idx)
            min_cost = c
    
    # Sort the remaining indices in ascending order
    remaining.sort()
    
    print(len(remaining))
    print(' '.join(map(str, remaining)))

if __name__ == "__main__":
    main()