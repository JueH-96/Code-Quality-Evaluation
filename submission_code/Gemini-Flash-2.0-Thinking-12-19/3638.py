from collections import Counter

class Solution:
    def makeStringGood(self, s: str) -> int:
        min_ops = float('inf')
        n = len(s)
        for target_freq in range(1, n + 1):
            counts = Counter(s)
            needed_decrease = {}
            needed_increase = {}
            for char_code in range(ord('a'), ord('z') + 1):
                char = chr(char_code)
                count = counts.get(char, 0)
                if count > target_freq:
                    needed_decrease[char] = count - target_freq
                elif count < target_freq:
                    needed_increase[char] = target_freq - count
                    
            deletions = sum(needed_decrease.values()) if needed_decrease else 0
            insertions = sum(needed_increase.values()) if needed_increase else 0
            changes = 0
            
            current_ops = deletions + insertions
            min_ops_for_freq = current_ops
            
            temp_needed_increase = needed_increase.copy()
            temp_needed_decrease = needed_decrease.copy()
            change_ops_count = 0
            
            for char_code in range(ord('a'), ord('z')):
                char = chr(char_code)
                next_char = chr(char_code + 1)
                if char in temp_needed_decrease and next_char in temp_needed_increase:
                    transfer_count = min(temp_needed_decrease[char], temp_needed_increase[next_char])
                    change_ops_count += transfer_count
                    temp_needed_decrease[char] -= transfer_count
                    temp_needed_increase[next_char] -= transfer_count
                    
            remaining_deletions = sum(v for v in temp_needed_decrease.values() if v > 0)
            remaining_insertions = sum(v for v in temp_needed_increase.values() if v > 0)
            
            current_ops_with_change = change_ops_count + remaining_deletions + remaining_insertions
            min_ops_for_freq = min(min_ops_for_freq, current_ops_with_change)
            min_ops = min(min_ops, min_ops_for_freq)
            
        return min_ops if min_ops != float('inf') else 0