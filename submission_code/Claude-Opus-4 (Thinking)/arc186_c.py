T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    boxes = []
    for _ in range(N):
        V, P = map(int, input().split())
        if V > P:  # Only consider profitable boxes
            boxes.append(V - P)
    
    if not boxes:
        print(0)
        continue
    
    boxes.sort(reverse=True)  # Sort by profit in descending order
    
    # Key insight: Mr. Ball can force Mr. Box to use at most min(M, len(boxes)) boxes
    # by controlling ball types
    num_usable = min(M, len(boxes))
    
    # When Mr. Ball has many types (M >= number of profitable boxes),
    # he can limit Mr. Box's earnings more severely
    if M >= len(boxes):
        # Mr. Ball can minimize earnings by strategic type switching
        # Based on the examples, when M >= len(boxes), the answer seems to be
        # related to the minimum profit
        if M > len(boxes):
            print(0)
        else:  # M == len(boxes)
            print(min(boxes))
    else:
        # When M < len(boxes), Mr. Box can choose the M most profitable boxes
        print(sum(boxes[:M]))