class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        if numFriends == 1:
            return word
        
        max_string = ""
        
        splits = []
        
        def generate_splits(index, current_split, count):
            if count == numFriends - 1:
                remaining_string = word[index:]
                if remaining_string:
                    new_split = current_split + [remaining_string]
                    splits.append(new_split)
                return
            
            for i in range(index, n - (numFriends - 1 - count)):
                new_split = current_split + [word[index:i+1]]
                generate_splits(i+1, new_split, count + 1)
        
        generate_splits(0, [], 0)
        
        for split in splits:
            for s in split:
                if s > max_string:
                    max_string = s
        
        return max_string