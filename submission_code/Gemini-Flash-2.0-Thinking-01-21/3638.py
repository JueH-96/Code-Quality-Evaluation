from collections import Counter

class Solution:
    def makeStringGood(self, s: str) -> int:
        min_operations = float('inf')
        n = len(s)
        for target_freq in range(1, n + 1):
            counts = Counter(s)
            current_operations = 0
            
            # Deletions
            current_counts = counts.copy()
            for char in list(counts.keys()):
                if counts[char] > target_freq:
                    del_count = counts[char] - target_freq
                    current_operations += del_count
                    current_counts[char] -= del_count
                    
            # Changes and Insertions
            change_insert_ops = 0
            temp_counts = current_counts.copy()
            change_ops = 0
            
            possible_change = True
            while possible_change:
                possible_change = False
                deficit_chars = []
                excess_chars = []
                for char_code in range(ord('a'), ord('z') + 1):
                    char = chr(char_code)
                    if temp_counts[char] < target_freq:
                        deficit_chars.append(char)
                    elif temp_counts[char] > target_freq:
                        excess_chars.append(char)
                        
                for excess_char in excess_chars:
                    next_char = chr(ord(excess_char) + 1)
                    if next_char <= 'z' and next_char in deficit_chars:
                        change_ops += 1
                        temp_counts[excess_char] -= 1
                        temp_counts[next_char] += 1
                        possible_change = True
                        break
                if not possible_change:
                    break
                    
            insertion_ops = 0
            for char in temp_counts:
                if temp_counts[char] < target_freq:
                    insertion_ops += target_freq - temp_counts[char]
                    
            current_ops = current_operations + change_ops + insertion_ops
            min_operations = min(min_operations, current_ops)
            
        return min_operations