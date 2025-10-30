# YOUR CODE HERE
import sys
import heapq

def min_dissatisfaction(N, K, X, T):
    # Min-heap to store the next possible shipping days
    next_possible_days = []
    heapq.heappush(next_possible_days, 1)
    
    # Total dissatisfaction
    total_dissatisfaction = 0
    
    for i in range(N):
        # Get the earliest possible shipping day
        earliest_day = heapq.heappop(next_possible_days)
        
        # Calculate the shipping day for the current order
        shipping_day = max(earliest_day, T[i])
        
        # Calculate dissatisfaction for the current order
        dissatisfaction = shipping_day - T[i]
        total_dissatisfaction += dissatisfaction
        
        # Schedule the next possible shipping day
        heapq.heappush(next_possible_days, shipping_day + X)
        
        # If we can ship more orders together, add the same day again
        if (i + 1) % K != 0:
            heapq.heappush(next_possible_days, shipping_day)
    
    return total_dissatisfaction

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    T = list(map(int, data[3:]))
    
    result = min_dissatisfaction(N, K, X, T)
    print(result)