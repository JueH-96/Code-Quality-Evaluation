def min_cost_to_buy_boxes(N, M, A, B):
    # Pair the price and candy count, then sort by candy count
    boxes = sorted(zip(A, A), key=lambda x: x[1])  # (price, candy)
    # Sort B in descending order to match the most demanding person first
    B.sort(reverse=True)

    total_cost = 0
    selected_boxes = []

    for b in B:
        # Find the cheapest box that has at least b candies
        found = False
        for i in range(len(boxes)):
            if boxes[i][1] >= b:
                total_cost += boxes[i][0]
                selected_boxes.append(boxes[i])
                # Remove the selected box from the list
                boxes.pop(i)
                found = True
                break
        if not found:
            return -1  # Not enough boxes to satisfy the conditions

    return total_cost

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:N+2+M]))

result = min_cost_to_buy_boxes(N, M, A, B)
print(result)