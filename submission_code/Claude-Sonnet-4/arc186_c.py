def solve():
    n, m = map(int, input().split())
    boxes = []
    for _ in range(n):
        v, p = map(int, input().split())
        if v > p:  # Only consider profitable boxes
            boxes.append(v - p)
    
    if not boxes:
        return 0
    
    # Sort by profit in descending order
    boxes.sort(reverse=True)
    
    # Mr. Box can use up to M different boxes (one per ball type)
    # He'll choose the most profitable ones
    total_profit = sum(boxes[:min(len(boxes), m)])
    
    return total_profit

t = int(input())
for _ in range(t):
    print(solve())