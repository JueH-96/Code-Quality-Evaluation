class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        original_count = s.count('1')
        augmented = '1' + s + '1'
        
        # Parse the augmented string into runs of '0's and '1's
        runs = []
        current_char = None
        current_length = 0
        for c in augmented:
            if c == current_char:
                current_length += 1
            else:
                if current_char is not None:
                    runs.append((current_char, current_length))
                current_char = c
                current_length = 1
        runs.append((current_char, current_length))
        
        max_gain = 0
        has_candidate = False
        n = len(runs)
        
        for i in range(1, n - 1):
            if (runs[i][0] == '1' and
                runs[i-1][0] == '0' and
                runs[i+1][0] == '0'):
                left0 = runs[i-1][1]
                right0 = runs[i+1][1]
                current_gain = left0 + right0
                if current_gain > max_gain:
                    max_gain = current_gain
                has_candidate = True
        
        if has_candidate:
            return original_count + max_gain
        else:
            return original_count