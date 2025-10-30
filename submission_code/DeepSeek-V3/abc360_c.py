# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    W = list(map(int, data[N+1:2*N+1]))
    
    # Create a dictionary to map each box to its items
    box_to_items = defaultdict(list)
    for i in range(N):
        box_to_items[A[i]].append(i)
    
    # For each box, if it has more than one item, we need to move all but one
    total_cost = 0
    for box in box_to_items:
        items = box_to_items[box]
        if len(items) > 1:
            # We need to move all but the lightest item
            # So we sort the items by weight and keep the lightest
            items_sorted = sorted(items, key=lambda x: W[x])
            for i in range(1, len(items_sorted)):
                total_cost += W[items_sorted[i]]
    
    print(total_cost)

if __name__ == "__main__":
    main()