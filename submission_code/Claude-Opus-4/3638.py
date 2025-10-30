class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        
        # Count frequency of each character
        freq = Counter(s)
        
        # Get all unique frequencies that we might want to target
        # We should consider frequencies from 0 to max frequency in the string
        max_freq = max(freq.values())
        
        min_ops = float('inf')
        
        # Try each possible target frequency
        for target in range(max_freq + 1):
            if target == 0:
                # If target is 0, we need to delete all characters
                min_ops = min(min_ops, len(s))
                continue
            
            ops = 0
            surplus = []  # Characters with more than target occurrences
            deficit = []  # Characters with less than target occurrences
            
            # Check all 26 lowercase letters
            for c in 'abcdefghijklmnopqrstuvwxyz':
                current = freq.get(c, 0)
                if current > target:
                    surplus.append((c, current - target))
                elif current < target:
                    deficit.append((c, target - current))
            
            # Calculate operations needed
            # We can use changes to move surplus to adjacent deficit characters
            total_surplus = sum(count for _, count in surplus)
            total_deficit = sum(count for _, count in deficit)
            
            # Try to match surplus with deficit using changes
            changes = 0
            surplus_dict = dict(surplus)
            deficit_dict = dict(deficit)
            
            # For each surplus character, try to change to next character if it has deficit
            for c in 'abcdefghijklmnopqrstuvwxy':  # Can't change 'z'
                if c in surplus_dict and chr(ord(c) + 1) in deficit_dict:
                    # We can change character c to c+1
                    change_count = min(surplus_dict[c], deficit_dict[chr(ord(c) + 1)])
                    changes += change_count
                    surplus_dict[c] -= change_count
                    deficit_dict[chr(ord(c) + 1)] -= change_count
            
            # Remaining surplus needs to be deleted
            deletions = sum(surplus_dict.values())
            
            # Remaining deficit needs to be inserted
            insertions = sum(deficit_dict.values())
            
            ops = changes + deletions + insertions
            min_ops = min(min_ops, ops)
        
        return min_ops