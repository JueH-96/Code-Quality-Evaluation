class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        if not s:
            return 0
        
        # Swap '0' and '1' in the string for simplicity in processing
        s = list(s)
        for index, element in enumerate(s):
            if element == '0':
                s[index] = '1'
            else:
                s[index] = '0'
        s = ''.join(s)
        
        max_count, left, current_ones = 0, 0, 0
        for right in range(len(s)):
            if s[right] == '1':
                current_ones += 1
            
            # Reduce current_ones if window exceeds the allowed difference (numOps + 1)
            while current_ones > numOps + 1:
                if s[left] == '1':
                    current_ones -= 1
                left += 1
            
            # Update the maximum size of the sliding window after adjustment
            max_count = max(max_count, right - left + 1)
        
        # The answer is the length of the string minus the maximum size of the sliding window
        answer = len(s) - max_count

        return answer