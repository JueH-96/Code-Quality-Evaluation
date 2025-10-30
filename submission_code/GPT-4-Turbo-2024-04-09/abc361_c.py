def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # We need to minimize the range (max - min) of the remaining array after removing K elements.
    # A brute force approach would be infeasible due to the constraints, so we need an efficient strategy.
    
    # One efficient approach is to use a sliding window of size N-K (the size of the remaining array B).
    # We will calculate the difference between the maximum and minimum in each window of size N-K.
    
    # To efficiently get the max and min in a sliding window, we can use two deques:
    # - One to maintain the indices of the minimum values (monotonically increasing deque)
    # - One to maintain the indices of the maximum values (monotonically decreasing deque)
    
    from collections import deque
    
    def min_in_window(arr, window_size):
        min_deque = deque()
        max_deque = deque()
        min_range = float('inf')
        
        for i in range(N):
            # Remove elements not within the window
            if min_deque and min_deque[0] <= i - window_size:
                min_deque.popleft()
            if max_deque and max_deque[0] <= i - window_size:
                max_deque.popleft()
            
            # Maintain the deque for minimums
            while min_deque and arr[min_deque[-1]] >= arr[i]:
                min_deque.pop()
            min_deque.append(i)
            
            # Maintain the deque for maximums
            while max_deque and arr[max_deque[-1]] <= arr[i]:
                max_deque.pop()
            max_deque.append(i)
            
            # Calculate the range if we have filled at least one full window
            if i >= window_size - 1:
                current_min = arr[min_deque[0]]
                current_max = arr[max_deque[0]]
                min_range = min(min_range, current_max - current_min)
        
        return min_range
    
    # We need a window of size N-K
    result = min_in_window(A, N-K)
    print(result)

if __name__ == "__main__":
    main()