class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        if numFriends == 1:
            return word
        if numFriends == n:
            return word[0]
        
        max_str = ""
        
        def generate_splits(index, current_split, splits):
            nonlocal max_str
            if len(splits) == numFriends - 1:
                remaining = word[index:]
                splits.append(remaining)
                
                max_in_round = ""
                for s in splits:
                    if s > max_in_round:
                        max_in_round = s
                
                if max_in_round > max_str:
                    max_str = max_in_round
                
                splits.pop()
                return
            
            for i in range(index + 1, n):
                splits.append(word[index:i])
                generate_splits(i, current_split + [word[index:i]], splits)
                splits.pop()
        
        generate_splits(0, [], [])
        
        return max_str