import sys
import collections

def main():
    import sys
    import collections
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    
    freq = collections.defaultdict(int)
    min_deque = collections.deque()
    max_deque = collections.deque()
    
    left = 0
    min_diff = float('inf')
    
    for right in range(N):
        current = P[right]
        freq[current] += 1
        
        # Maintain min_deque
        while min_deque and P[right] < P[min_deque[-1]]:
            min_deque.pop()
        min_deque.append(right)
        
        # Maintain max_deque
        while max_deque and P[right] > P[max_deque[-1]]:
            max_deque.pop()
        max_deque.append(right)
        
        current_min = P[min_deque[0]]
        current_max = P[max_deque[0]]
        
        while current_max - current_min + 1 >= K and len(freq) >= K:
            if current_max - current_min + 1 == K and len(freq) == K:
                min_diff = min(min_diff, right - left)
            # Remove element at left
            left_element = P[left]
            freq[left_element] -= 1
            if freq[left_element] == 0:
                del freq[left_element]
            # Remove left index from deques if necessary
            if left == min_deque[0]:
                min_deque.popleft()
            if left == max_deque[0]:
                max_deque.popleft()
            left += 1
            # Update current min and max
            if min_deque:
                current_min = P[min_deque[0]]
                current_max = P[max_deque[0]]
            else:
                break
    
    print(min_diff)

if __name__ == "__main__":
    main()