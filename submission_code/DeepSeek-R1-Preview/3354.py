class Solution:
    def minimizeStringValue(self, s: str) -> str:
        counts = [0] * 26  # To keep track of counts of each character
        s_list = list(s)    # Convert string to list for easy manipulation
        
        for i in range(len(s_list)):
            if s_list[i] == '?':
                # Create a copy of current counts to evaluate the best character
                current_counts = counts.copy()
                min_count = min(current_counts)
                
                # Collect all characters with the minimal count
                candidates = []
                for c in range(26):
                    if current_counts[c] == min_count:
                        candidates.append(c)
                
                # Choose the lex smallest character among candidates
                chosen_char = chr(ord('a') + candidates[0])
                s_list[i] = chosen_char
                
                # Update the count for the chosen character
                counts[candidates[0]] += 1
            else:
                # Update the count for the current character
                counts[ord(s_list[i]) - ord('a')] += 1
        
        return ''.join(s_list)