# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Create list of boxes with their indices for tracking
boxes = [(A[i], i) for i in range(N)]
boxes.sort()  # Sort by candy count (which equals price)

# Sort people by their requirements in descending order
people = [(B[i], i) for i in range(M)]
people.sort(reverse=True)

used_boxes = set()
total_cost = 0

# For each person (most demanding first)
for requirement, person_id in people:
    found = False
    # Find the cheapest available box that satisfies this person's requirement
    for candy_count, box_id in boxes:
        if box_id not in used_boxes and candy_count >= requirement:
            used_boxes.add(box_id)
            total_cost += candy_count  # cost equals candy count
            found = True
            break
    
    if not found:
        print(-1)
        exit()

print(total_cost)