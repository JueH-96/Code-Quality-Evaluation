def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    cards = []
    idx = 1
    # Read the inputs
    for i in range(N):
        A = int(data[idx])
        C = int(data[idx+1])
        idx += 2
        # Store (A, C, index)
        cards.append((A, C, i+1))
    
    # Sort by A descending
    cards.sort(key=lambda x: x[0], reverse=True)

    remaining = []
    min_cost = float('inf')
    
    # Greedily keep each card whose cost is smaller
    # than any cost encountered so far among those
    # with a strictly larger A
    for A, C, i in cards:
        if C < min_cost:
            remaining.append(i)
            min_cost = C
    
    # Sort by card index in ascending order
    remaining.sort()
    
    # Output
    print(len(remaining))
    print(*remaining)

def main():
    solve()

if __name__ == "__main__":
    main()