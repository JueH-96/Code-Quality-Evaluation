import sys
import heapq

def main():
    data = sys.stdin.read().split()
    ptr = 0
    Q = int(data[ptr])
    ptr += 1
    current_time = 0
    heap = []
    
    for _ in range(Q):
        query_type = int(data[ptr])
        ptr += 1
        if query_type == 1:
            # Add a plant with height 0
            heapq.heappush(heap, current_time)
        elif query_type == 2:
            # Wait for T days
            T = int(data[ptr])
            ptr += 1
            current_time += T
        elif query_type == 3:
            # Harvest plants with height at least H
            H = int(data[ptr])
            ptr += 1
            H_threshold = current_time - H
            count = 0
            while heap and heap[0] <= H_threshold:
                heapq.heappop(heap)
                count += 1
            print(count)

if __name__ == '__main__':
    main()