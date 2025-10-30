from itertools import combinations

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    b.sort()

    boxes = []
    for price in a:
        boxes.append((price, price))

    min_cost = float('inf')
    found_solution = False

    for indices in combinations(range(n), m):
        selected_boxes_with_prices = []
        for i in indices:
            selected_boxes_with_prices.append((boxes[i][1], boxes[i][0]))  # (candies, price)

        selected_boxes_with_prices.sort()

        feasible = True
        for i in range(m):
            if selected_boxes_with_prices[i][0] < b[i]:
                feasible = False
                break

        if feasible:
            current_cost = sum(box[1] for box in selected_boxes_with_prices)
            min_cost = min(min_cost, current_cost)
            found_solution = True

    if found_solution:
        print(min_cost)
    else:
        print(-1)

if __name__ == "__main__":
    solve()