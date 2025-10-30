class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        from collections import deque
        
        # Find all starting indices of substring a in s
        indices_a = [i for i in range(len(s) - len(a) + 1) if s[i:i+len(a)] == a]
        # Find all starting indices of substring b in s
        indices_b = [i for i in range(len(s) - len(b) + 1) if s[i:i+len(b)] == b]
        
        # Result list for beautiful indices
        beautiful_indices = []
        
        # Use two pointers to find valid indices
        j = 0
        queue_b = deque(indices_b)  # Use deque to efficiently manage the window of valid b indices
        
        for i in indices_a:
            # Move the left boundary of the queue to maintain the distance condition
            while queue_b and queue_b[0] < i - k:
                queue_b.popleft()
            
            # Check if there's any valid b index in the range [i-k, i+k]
            while queue_b and queue_b[0] <= i + k:
                if abs(queue_b[0] - i) <= k:
                    beautiful_indices.append(i)
                    break
                queue_b.popleft()
        
        return beautiful_indices