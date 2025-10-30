def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))
    
    # We need to find the minimum i_K - i_1 for a valid sequence
    # We can use a sliding window approach to find all valid subsequences of length K
    
    from collections import deque
    
    # This will store the indices of the elements in the current window
    # We need to check if the elements in this window can form a sequence of K consecutive numbers
    min_diff = float('inf')
    
    # We need to track the positions of each number in the permutation
    positions = [0] * (N + 1)
    for index, value in enumerate(P):
        positions[value] = index
    
    # We will use a deque to maintain the indices in the current window
    window = deque()
    
    # We iterate over the permutation and maintain a window of size K
    for i in range(N):
        # Add the current index to the window
        while window and window[0] <= i - K:
            window.popleft()
        
        # Maintain the window sorted by the value of P (we need the smallest range of indices)
        while window and P[window[-1]] > P[i]:
            window.pop()
        
        window.append(i)
        
        # Check if the window has size K
        if len(window) == K:
            # Check if the elements in the window can form a sequence of K consecutive numbers
            # We need to check if the max element - min element + 1 == K
            sorted_window = sorted(P[idx] for idx in window)
            if sorted_window[-1] - sorted_window[0] + 1 == K:
                # Calculate i_K - i_1
                current_diff = window[-1] - window[0]
                min_diff = min(min_diff, current_diff)
    
    print(min_diff)