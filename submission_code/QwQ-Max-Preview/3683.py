class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        
        K = numFriends
        len_word = len(word)
        max_char = max(word)
        candidates = []
        
        for a in range(len_word):
            if word[a] != max_char:
                continue
            
            possible_b = []
            # Check if it can be the first part
            if a == 0:
                max_b_first = len_word - K
                if max_b_first >= 0:
                    possible_b.append(max_b_first)
            # Check if it can be the last part
            if a >= K - 1:
                possible_b.append(len_word - 1)
            # Check if it can be a middle part (K >=3 and a >=1)
            if K >= 3 and a >= 1:
                max_b_middle = len_word - K
                if max_b_middle >= a:
                    possible_b.append(max_b_middle)
            
            # Collect all possible substrings for this a
            for b in possible_b:
                substring = word[a : b + 1]
                candidates.append(substring)
        
        return max(candidates) if candidates else ''