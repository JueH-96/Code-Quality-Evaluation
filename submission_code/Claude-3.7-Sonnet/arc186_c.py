def solve_game(boxes, M):
    # Sort boxes by profitability (Vi - Pi)
    boxes.sort(key=lambda box: box[0] - box[1], reverse=True)
    
    total_profit = 0
    boxes_used = 0
    
    for capacity, cost in boxes:
        profit = min(capacity - cost, 1)  # Profit is at most 1 per box
        if profit <= 0:  # Only consider profitable boxes
            continue
        
        total_profit += profit
        boxes_used += 1
        
        if boxes_used >= M:  # Can't use more than M boxes (limited by ball types)
            break
    
    return total_profit

def main():
    T = int(input())
    for _ in range(T):
        N, M = map(int, input().split())
        boxes = []
        for _ in range(N):
            V, P = map(int, input().split())
            boxes.append((V, P))
        
        print(solve_game(boxes, M))

if __name__ == "__main__":
    main()