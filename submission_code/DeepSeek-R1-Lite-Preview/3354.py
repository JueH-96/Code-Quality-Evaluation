import heapq
from collections import defaultdict

class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Step 1: Count initial frequencies of each character (non '?')
        freq = defaultdict(int)
        for char in s:
            if char != '?':
                freq[char] += 1
        
        # Step 2: Prepare min-heap with initial frequencies
        heap = []
        for char in 'abcdefghijklmnopqrstuvwxyz':
            heap.append((freq[char], char))
        heapq.heapify(heap)
        
        k = s.count('?')
        
        # Step 3: Distribute k additions using the min-heap
        for _ in range(k):
            current_freq, char = heapq.heappop(heap)
            current_freq += 1
            heapq.heappush(heap, (current_freq, char))
        
        # Step 4: Collect the characters assigned to '?'
        assigned_chars = []
        for _ in range(26):
            current_freq, char = heapq.heappop(heap)
            initial_freq = freq[char]
            additions = current_freq - initial_freq
            assigned_chars.extend([char] * additions)
        
        # Step 5: Sort the assigned characters to get lex smallest sequence
        assigned_chars.sort()
        
        # Step 6: Replace '?' in the original string with the sorted assigned characters
        result = []
        assigned_idx = 0
        for char in s:
            if char == '?':
                result.append(assigned_chars[assigned_idx])
                assigned_idx += 1
            else:
                result.append(char)
        
        return ''.join(result)