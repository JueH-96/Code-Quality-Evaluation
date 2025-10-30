import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Compute prefix array
    prefix = [0] * N
    seen = set()
    for i in range(N):
        seen.add(A[i])
        prefix[i] = len(seen)
    
    # Compute suffix array
    suffix = [0] * N
    seen = set()
    for i in range(N-1, -1, -1):
        seen.add(A[i])
        suffix[i] = len(seen)
    
    # Process to find maximum contribution
    last_occurrence = {}
    start = 1  # left starts from 1
    dq = deque()
    answer = 0
    
    for j in range(1, N-1):  # j goes up to N-2 (inclusive)
        current = A[j]
        if current in last_occurrence:
            start = max(start, last_occurrence[current] + 1)
        last_occurrence[current] = j
        
        # current_left is j
        current_left = j
        adj = prefix[current_left - 1] - current_left
        
        # Maintain deque for the current window [start..j]
        while dq and (prefix[dq[-1] - 1] - dq[-1]) <= adj:
            dq.pop()
        dq.append(current_left)
        
        # Remove elements from front that are out of the window
        while dq[0] < start:
            dq.popleft()
        
        # Calculate contribution if deque is not empty
        if dq:
            max_adj_val = prefix[dq[0] - 1] - dq[0]
            contribution = max_adj_val + j + 1
            total = contribution + suffix[j + 1]
            if total > answer:
                answer = total
    
    print(answer)

if __name__ == "__main__":
    main()