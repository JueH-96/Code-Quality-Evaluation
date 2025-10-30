N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Check if any person's requirement exceeds all available boxes
possible = True
for b in B:
    if all(a < b for a in A):
        possible = False
        break

if not possible:
    print(-1)
else:
    # For each person i, find valid boxes that have enough candy
    valid_boxes = []
    for b in B:
        boxes = []
        for j in range(N):
            if A[j] >= b:
                boxes.append((A[j], j, A[j]))  # (candy, box_index, price)
        valid_boxes.append(boxes)
    
    # Try all possible combinations using backtracking
    def solve(person, used_boxes, total_cost):
        if person == M:
            return total_cost
            
        min_cost = float('inf')
        for candy, box_idx, price in valid_boxes[person]:
            if box_idx not in used_boxes:
                used_boxes.add(box_idx)
                cost = solve(person + 1, used_boxes, total_cost + price)
                used_boxes.remove(box_idx)
                if cost < min_cost:
                    min_cost = cost
                    
        return min_cost
    
    result = solve(0, set(), 0)
    print(result)