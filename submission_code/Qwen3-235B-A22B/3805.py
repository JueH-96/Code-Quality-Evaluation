class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        augmented_s = '1' + s + '1'
        blocks = []
        if not augmented_s:
            return 0  # should not happen
        
        current_char = augmented_s[0]
        count = 1
        for c in augmented_s[1:]:
            if c == current_char:
                count += 1
            else:
                blocks.append((current_char, count))
                current_char = c
                count = 1
        blocks.append((current_char, count))
        
        surroundedAs = []  # each entry is (a_length, left, right)
        surroundedBs = []
        
        for i in range(len(blocks)):
            if blocks[i][0] == '0':
                # Check if surrounded by '1's for surrounded B
                if i > 0 and i < len(blocks) - 1:
                    if blocks[i-1][0] == '1' and blocks[i+1][0] == '1':
                        surroundedBs.append(blocks[i][1])
            else:
                # Check if surrounded by '0's for surrounded A
                if i > 0 and i < len(blocks) - 1:
                    if blocks[i-1][0] == '0' and blocks[i+1][0] == '0':
                        a_length = blocks[i][1]
                        left = blocks[i-1][1]
                        right = blocks[i+1][1]
                        surroundedAs.append((a_length, left, right))
        
        original_count = s.count('1')
        
        gain_4a_candidate = -float('inf')
        gain_4b_candidate = -float('inf')
        
        # Compute gain_4a_candidate
        if surroundedAs and surroundedBs:
            min_a = min(a[0] for a in surroundedAs)
            max_b = max(surroundedBs)
            gain_4a_candidate = max_b - min_a
        
        # Compute gain_4b_candidate
        if surroundedAs:
            max_sum = max(left + right for (a, left, right) in surroundedAs)
            gain_4b_candidate = max_sum
        
        # Determine max_gain
        max_gain = max(gain_4a_candidate, gain_4b_candidate, 0)
        
        return original_count + max_gain