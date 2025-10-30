import sys

def solve():
    """
    Reads input, solves the problem, and prints the output to STDOUT.
    """
    # Use fast I/O
    input = sys.stdin.readline

    try:
        # Read the number of boxes and items
        n_str = input()
        if not n_str:
            return
        N = int(n_str)
        
        # Read the initial box locations for each item
        A = list(map(int, input().split()))
        
        # Read the weights of each item
        W = list(map(int, input().split()))
    except (ValueError, IndexError):
        # Handle cases with malformed or empty input
        return

    # `boxes` is a list of lists. `boxes[i]` will store the weights of items in box i+1.
    boxes = [[] for _ in range(N)]
    
    # Group item weights by their box number.
    for i in range(N):
        # The input A is 1-indexed, so we use A[i] - 1 for 0-indexed list access.
        box_index = A[i] - 1
        boxes[box_index].append(W[i])
        
    total_cost = 0
    
    # Iterate through each box's content.
    for weight_list in boxes:
        # A cost is only incurred for boxes that are over-occupied (contain more than one item).
        if len(weight_list) > 1:
            # To leave one item, we must move all others.
            # To minimize cost, we keep the heaviest item and move the rest.
            # The cost for this box is the sum of weights of items moved.
            cost_for_this_box = sum(weight_list) - max(weight_list)
            total_cost += cost_for_this_box
            
    # Print the minimum total cost.
    print(total_cost)

solve()