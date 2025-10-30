import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))

    # box_items[j] will store a list of weights of items initially in box j.
    # Using 1-based indexing for boxes (1 to N), so size N+1.
    box_items = [[] for _ in range(N + 1)]

    # Populate box_items with item weights based on their initial box.
    for i in range(N):
        box_items[A[i]].append(W[i])

    total_cost = 0

    # Iterate through each box to calculate the minimum cost.
    # The strategy is to always leave the heaviest item in a surplus box,
    # and move the lighter ones out. These moved items contribute to the cost.
    # The number of such items will exactly match the number of empty boxes,
    # so they can fill those empty boxes.
    for j in range(1, N + 1):
        if len(box_items[j]) > 1:
            # This is a surplus box (contains more than one item).
            # To minimize cost, we want to maximize the weight of the item that stays.
            # So, we sort the items by weight and keep the heaviest one in this box.
            # The remaining (lighter) items must move out, and their weights contribute to the total cost.
            box_items[j].sort()  # Sort weights in ascending order
            
            # Sum the weights of all items except the heaviest one.
            # These are the items that are forced to move from this box.
            for k in range(len(box_items[j]) - 1):
                total_cost += box_items[j][k]
        
        # If len(box_items[j]) == 0 (empty box) or len(box_items[j]) == 1 (perfect box):
        # These boxes do not directly add to the cost in this step.
        # An empty box will receive an item, but the cost for that item's movement is accounted for
        # at its original location (if it was one of the 'forced to move' items from a surplus box).
        # An item in a perfect box can stay, contributing 0 cost.

    print(total_cost)

# Call the solve function to execute the program.
solve()