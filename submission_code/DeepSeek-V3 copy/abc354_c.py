def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = []
    C = []
    index = []
    for i in range(N):
        A.append(int(data[1 + 2*i]))
        C.append(int(data[2 + 2*i]))
        index.append(i + 1)
    
    # Combine A, C, and index into a list of tuples and sort by A in ascending order
    cards = list(zip(A, C, index))
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