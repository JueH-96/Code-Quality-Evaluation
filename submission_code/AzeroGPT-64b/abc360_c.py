from collections import defaultdict
import sys

def main():
    # Start by reading the input value of N
    N = int(input())

    # Read each item's current box position A_i and weight W_i
    A = list(map(int, input().split()))
    W = list(map(int, input().split()))

    # Create a dictionary to store the lowest weight of an item for each box
    min_weight_per_box = defaultdict(lambda: float('inf'))

    # Populate the dictionary with the current minimum weights for each box
    for i, box in enumerate(A, 1):
        min_weight_per_box[box] = min(min_weight_per_box[box], W[i - 1])

    # Create a list to store the total weight of items in each box
    total_box_weight = [0] * (N + 1)

    # Update the total weight for items in each box
    for i, box in enumerate(A, 1):
        total_box_weight[box] += W[i - 1]

    # Calculate the cost of moving items to their target boxes
    total_cost = 0
    for box in range(1, N + 1):
        # If a box has more than one item, calculate the cost to remove extra items
        if total_box_weight[box] > min_weight_per_box[box]:
            total_cost += total_box_weight[box] - min_weight_per_box[box]

    # Output the minimum cost required
    print(total_cost)

if __name__ == "__main__":
    main()