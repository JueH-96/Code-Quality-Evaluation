from sys import stdin

def read_ints():
    return map(int, stdin.readline().split())

def read_int():
    return int(stdin.readline())

def main():
    n = read_int()
    a = list(read_ints())
    w = list(read_ints())

    # Create a dictionary to store the items in each box
    boxes = {i: [] for i in range(1, n + 1)}
    for i, item in enumerate(a, start=1):
        boxes[item].append((i, w[i - 1]))

    # Initialize the total cost
    total_cost = 0

    # Iterate over the boxes
    for box in range(1, n + 1):
        # If the box is empty, find the item that should be in this box
        if not boxes[box]:
            for item in range(1, n + 1):
                if item not in boxes:
                    # Move the item to the empty box
                    total_cost += w[item - 1]
                    boxes[box].append((item, w[item - 1]))
                    break
        # If the box has more than one item, move the items to their correct boxes
        elif len(boxes[box]) > 1:
            # Sort the items in the box by their weights
            boxes[box].sort(key=lambda x: x[1])
            # Move the items to their correct boxes
            for item, weight in boxes[box][1:]:
                total_cost += weight
                boxes[item].append((item, weight))

    print(total_cost)

if __name__ == "__main__":
    main()