import heapq
import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    K = int(data[1])
    Q = int(data[2])
    
    updates = []
    index = 3
    for _ in range(Q):
        X = int(data[index]) - 1  # Convert to 0-based index
        Y = int(data[index + 1])
        updates.append((X, Y))
        index += 2
    
    A = [0] * N
    top_k = []
    current_sum = 0
    
    for X, Y in updates:
        old_value = A[X]
        
        # If old_value is in top_k, remove it
        if old_value in top_k:
            top_k.remove(old_value)
            heapq.heapify(top_k)
            current_sum -= old_value
        
        # Update the value in A
        A[X] = Y
        
        # Add new value if it should be in top_k
        if len(top_k) < K:
            heapq.heappush(top_k, Y)
            current_sum += Y
        else:
            if Y > top_k[0]:
                current_sum += Y
                current_sum -= heapq.heappushpop(top_k, Y)
        
        print(current_sum)

if __name__ == "__main__":
    main()