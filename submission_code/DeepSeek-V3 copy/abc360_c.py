def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    W = list(map(int, data[N+1:2*N+1]))
    
    # Create a dictionary to map each box to its items
    box_to_items = {}
    for i in range(N):
        box = A[i]
        if box not in box_to_items:
            box_to_items[box] = []
        box_to_items[box].append((W[i], i+1))  # (weight, item number)
    
    # For each box that has more than one item, we need to move all but one
    total_cost = 0
    for box in box_to_items:
        items = box_to_items[box]
        if len(items) > 1:
            # Keep the item with the maximum weight in the box
            # Move the others
            max_weight = max(item[0] for item in items)
            for item in items:
                if item[0] != max_weight:
                    total_cost += item[0]
    
    print(total_cost)

if __name__ == "__main__":
    main()