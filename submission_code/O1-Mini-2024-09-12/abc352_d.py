from collections import deque
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N, K = int(data[0]), int(data[1])
    P = list(map(int, data[2:2+N]))
    
    pos = [0]*(N+1)
    for idx, val in enumerate(P):
        pos[val] = idx +1  # 1-based indexing

    pos_sorted = [pos[x] for x in range(1, N+1)]
    
    min_diff = float('inf')
    dq_min = deque()
    dq_max = deque()
    
    for i in range(len(pos_sorted)):
        # Maintain deque for min
        while dq_min and pos_sorted[i] <= pos_sorted[dq_min[-1]]:
            dq_min.pop()
        dq_min.append(i)
        
        # Maintain deque for max
        while dq_max and pos_sorted[i] >= pos_sorted[dq_max[-1]]:
            dq_max.pop()
        dq_max.append(i)
        
        # Remove elements out of window
        if i >= K:
            if dq_min[0] == i - K:
                dq_min.popleft()
            if dq_max[0] == i - K:
                dq_max.popleft()
        
        # Once we have the first window
        if i >= K-1:
            current_min = pos_sorted[dq_min[0]]
            current_max = pos_sorted[dq_max[0]]
            diff = current_max - current_min
            if diff < min_diff:
                min_diff = diff
                
    print(min_diff)

if __name__ == "__main__":
    main()