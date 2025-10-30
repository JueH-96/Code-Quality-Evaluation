class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        m, n = len(word2), len(word1)
        if m > n:
            return []
        
        # Precompute next occurrence of each character for each position
        next_char = [{} for _ in range(n + 1)]
        for c in set(word2):
            next_char[n][c] = -1
        for i in range(n-1, -1, -1):
            current_c = word1[i]
            next_char[i] = next_char[i+1].copy()
            next_char[i][current_c] = i
        
        # Function to check if the remaining part can be matched as a subsequence starting from start_j
        def is_subsequence(start_j, s):
            current_pos = start_j
            for c in s:
                if current_pos >= n:
                    return False
                next_j = next_char[current_pos].get(c, -1)
                if next_j == -1:
                    return False
                current_pos = next_j + 1
            return True
        
        sequence = []
        change_used = False
        
        for i in range(m):
            current_c = word2[i]
            prev_j = sequence[-1] if sequence else -1
            start_j = prev_j + 1
            if start_j >= n:
                return []
            
            # Try to find a matching character
            j = next_char[start_j].get(current_c, -1)
            if j != -1:
                sequence.append(j)
                continue
            
            # If no match, try to use a change
            if not change_used:
                remaining = word2[i+1:]
                required_length = len(remaining)
                max_j = n - required_length - 1
                if max_j < start_j:
                    return []
                found = False
                # Iterate possible j_candidate to find the earliest valid
                for j_candidate in range(start_j, max_j + 1):
                    if is_subsequence(j_candidate + 1, remaining):
                        # Check if the remaining can be matched and build the sequence
                        temp_sequence = sequence.copy()
                        temp_sequence.append(j_candidate)
                        current_pos = j_candidate + 1
                        valid = True
                        for c in remaining:
                            next_j = next_char[current_pos].get(c, -1)
                            if next_j == -1:
                                valid = False
                                break
                            temp_sequence.append(next_j)
                            current_pos = next_j + 1
                        if valid:
                            return temp_sequence
                # If no valid j_candidate found after checking all possibilities
                return []
            else:
                return []
        
        return sequence if len(sequence) == m else []