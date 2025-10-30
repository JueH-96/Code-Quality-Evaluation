class Solution:
    def minimumLength(self, s: str) -> int:
        # Count the occurrences of each character
        freq = [0]*26
        for ch in s:
            freq[ord(ch) - ord('a')] += 1
        
        # For each character frequency k:
        #  - if k == 0 => contributes 0
        #  - if k == 1 => contributes 1
        #  - if k == 2 => contributes 2
        #  - if k >= 3:
        #       if k is odd => contributes 1
        #       if k is even => contributes 2
        # Sum of these contributions gives the final minimal length.
        
        answer = 0
        for k in freq:
            if k == 0:
                continue
            elif k == 1:
                answer += 1
            elif k == 2:
                answer += 2
            else:
                answer += 1 if k % 2 else 2
        
        return answer