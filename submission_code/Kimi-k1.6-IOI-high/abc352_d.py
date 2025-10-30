from collections import deque
import sys

def main():
    n, k = map(int, sys.stdin.readline().split())
    p = list(map(int, sys.stdin.readline().split()))
    
    pos = [0] * (n + 1)  # pos[0] is unused
    for i in range(n):
        pos[p[i]] = i + 1  # 1-based position
    
    arr = [pos[i] for i in range(1, n+1)]
    
    min_deque = deque()
    max_deque = deque()
    answer = float('inf')
    
    for i in range(n):
        # Remove elements out of the current window from the front
        while min_deque and min_deque[0] < i - k + 1:
            min_deque.popleft()
        while max_deque and max_deque[0] < i - k + 1:
            max_deque.popleft()
        
        # Maintain min_deque
        while min_deque and arr[min_deque[-1]] >= arr[i]:
            min_deque.pop()
        min_deque.append(i)
        
        # Maintain max_deque
        while max_deque and arr[max_deque[-1]] <= arr[i]:
            max_deque.pop()
        max_deque.append(i)
        
        # Update answer when the window is valid
        if i >= k - 1:
            current_min = arr[min_deque[0]]
            current_max = arr[max_deque[0]]
            diff = current_max - current_min
            if diff < answer:
                answer = diff
    
    print(answer)

if __name__ == '__main__':
    main()