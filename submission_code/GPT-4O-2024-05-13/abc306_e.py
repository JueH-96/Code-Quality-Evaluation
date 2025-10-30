# YOUR CODE HERE
import heapq
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    Q = int(data[2])
    
    updates = []
    for i in range(Q):
        X = int(data[3 + 2 * i]) - 1
        Y = int(data[4 + 2 * i])
        updates.append((X, Y))
    
    A = [0] * N
    max_heap = []
    index_map = {}
    
    for i in range(N):
        heapq.heappush(max_heap, (-A[i], i))
        index_map[i] = -A[i]
    
    current_sum = 0
    for i in range(K):
        current_sum += -max_heap[i][0]
    
    results = []
    
    for X, Y in updates:
        old_value = -index_map[X]
        new_value = Y
        index_map[X] = -new_value
        
        if old_value != new_value:
            heapq.heappush(max_heap, (-new_value, X))
        
        while max_heap and index_map[max_heap[0][1]] != max_heap[0][0]:
            heapq.heappop(max_heap)
        
        current_sum = 0
        for i in range(K):
            current_sum += -max_heap[i][0]
        
        results.append(current_sum)
    
    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()