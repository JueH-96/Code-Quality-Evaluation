import heapq
import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    Q = int(data[idx])
    idx += 1
    heap = []
    current_total = 0
    for _ in range(Q):
        query_type = data[idx]
        if query_type == '1':
            heapq.heappush(heap, current_total)
            idx += 1
        elif query_type == '2':
            T = int(data[idx+1])
            current_total += T
            idx += 2
        else:
            H = int(data[idx+1])
            X = current_total - H
            count = 0
            while heap and heap[0] <= X:
                heapq.heappop(heap)
                count += 1
            print(count)
            idx += 2

if __name__ == '__main__':
    main()