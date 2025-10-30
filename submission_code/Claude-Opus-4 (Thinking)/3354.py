class Solution:
    def minimizeStringValue(self, s: str) -> str:
        from collections import defaultdict
        import heapq
        
        # Count fixed characters and find question mark positions
        char_count = defaultdict(int)
        question_positions = []
        
        for i, ch in enumerate(s):
            if ch == '?':
                question_positions.append(i)
            else:
                char_count[ch] += 1
        
        # Create a min heap with (count, character) for all 26 characters
        heap = []
        for ch in 'abcdefghijklmnopqrstuvwxyz':
            heapq.heappush(heap, (char_count[ch], ch))
        
        # Determine which characters to use for question marks
        chars_to_use = []
        for _ in question_positions:
            # Get the character with minimum count
            count, ch = heapq.heappop(heap)
            chars_to_use.append(ch)
            # Increment its count and push back
            heapq.heappush(heap, (count + 1, ch))
        
        # Sort characters to use for lexicographically smallest result
        chars_to_use.sort()
        
        # Build the result
        result = list(s)
        for i, pos in enumerate(question_positions):
            result[pos] = chars_to_use[i]
        
        return ''.join(result)