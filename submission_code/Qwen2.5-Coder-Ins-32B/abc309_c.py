# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    medicines = []
    
    index = 2
    for _ in range(N):
        a = int(data[index])
        b = int(data[index + 1])
        medicines.append((a, b))
        index += 2
    
    # Min-heap to keep track of the days when the medicine ends
    min_heap = []
    current_pills = 0
    day = 1
    
    while True:
        # Add new medicines that start on this day
        while medicines and medicines[-1][0] == day:
            a, b = medicines.pop()
            heapq.heappush(min_heap, (day + a - 1, b))
            current_pills += b
        
        # Remove medicines that have ended
        while min_heap and min_heap[0][0] < day:
            _, b = heapq.heappop(min_heap)
            current_pills -= b
        
        # Check if the current day's pill count is K or less
        if current_pills <= K:
            print(day)
            return
        
        # Move to the next day
        if medicines:
            day = min(day + 1, medicines[-1][0])
        elif min_heap:
            day = min_heap[0][0] + 1
        else:
            break

if __name__ == "__main__":
    main()