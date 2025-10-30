def solve():
    T = int(input())
    
    for _ in range(T):
        N, M = map(int, input().split())
        boxes = []
        
        for _ in range(N):
            V, P = map(int, input().split())
            profit = V - P
            if profit > 0:
                boxes.append(profit)
        
        # Sort in descending order
        boxes.sort(reverse=True)
        
        # Take at most M boxes with positive profit
        result = sum(boxes[:min(M, len(boxes))])
        
        print(result)

solve()