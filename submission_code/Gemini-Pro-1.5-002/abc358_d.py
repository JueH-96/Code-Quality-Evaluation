# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    boxes = []
    for i in range(n):
        boxes.append((a[i], i))
    
    boxes.sort()
    b.sort()

    min_cost = float('inf')
    
    for i in range(1 << n):
        chosen_boxes = []
        cost = 0
        count = 0
        for j in range(n):
            if (i >> j) & 1:
                chosen_boxes.append(boxes[j])
                cost += boxes[j][0]
                count += 1
        
        if count != m:
            continue

        
        possible = True
        
        chosen_boxes.sort()
        
        
        satisfied = [False] * m
        
        
        def backtrack(index):
            if index == m:
                return True
            
            for j in range(m):
                if not satisfied[j] and chosen_boxes[index][0] >= b[j]:
                    satisfied[j] = True
                    if backtrack(index + 1):
                        return True
                    satisfied[j] = False
            return False
        
        if backtrack(0):
            min_cost = min(min_cost, cost)

    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)

solve()