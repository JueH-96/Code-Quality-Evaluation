def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    W = list(map(int, data[N+1:2*N+1]))
    
    # Initialize list of lists for boxes
    boxes = [[] for _ in range(N+1)]
    
    # Populate the boxes with item weights
    for idx in range(N):
        box_number = A[idx]
        boxes[box_number].append(W[idx])
    
    total_cost = 0
    
    for box_number in range(1, N+1):
        items = boxes[box_number]
        if len(items) > 1:
            # Sort the weights in ascending order
            items_sorted = sorted(items)
            # Move the smallest (len(items) - 1) items
            total_cost += sum(items_sorted[:len(items)-1])
    
    print(total_cost)

if __name__ == '__main__':
    main()