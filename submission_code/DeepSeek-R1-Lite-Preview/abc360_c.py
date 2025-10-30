def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    W = list(map(int, data[N+1:2*N+1]))
    
    # Initialize list of lists for boxes
    boxes = [[] for _ in range(N+1)]
    
    for a, w in zip(A, W):
        boxes[a].append(w)
    
    total_cost = 0
    
    for i in range(1, N+1):
        items = boxes[i]
        if len(items) > 1:
            # Sort items in descending order
            sorted_items = sorted(items, reverse=True)
            # Keep the heaviest item and move the rest
            move_items = sorted_items[1:]
            total_cost += sum(move_items)
    
    print(total_cost)

if __name__ == "__main__":
    main()