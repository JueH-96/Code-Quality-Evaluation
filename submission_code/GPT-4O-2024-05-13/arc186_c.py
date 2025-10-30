# YOUR CODE HERE
import sys
input = sys.stdin.read

def solve():
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    
    for _ in range(T):
        N = int(data[idx])
        M = int(data[idx + 1])
        idx += 2
        
        boxes = []
        for _ in range(N):
            V = int(data[idx])
            P = int(data[idx + 1])
            boxes.append((V, P))
            idx += 2
        
        # Sort boxes by price
        boxes.sort(key=lambda x: x[1])
        
        total_cost = 0
        total_earnings = 0
        
        for V, P in boxes:
            total_cost += P
            total_earnings += V
            if total_earnings - total_cost < 0:
                break
        
        results.append(total_earnings - total_cost)
    
    for result in results:
        print(result)