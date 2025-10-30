class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        if numFriends == 1:
            return word
        if numFriends == n:
            return word[0]
        
        ans = ""
        
        def generate_splits(index, current_split):
            nonlocal ans
            if index == n:
                if len(current_split) == numFriends:
                    for s in current_split:
                        ans = max(ans, s)
                return

            if len(current_split) > numFriends:
                return

            if not current_split:
                generate_splits(index + 1, [word[index]])
            else:
                
                new_split = current_split[:-1] + [current_split[-1] + word[index]]
                generate_splits(index + 1, new_split)
                
                if len(current_split) < numFriends:
                    generate_splits(index+1, current_split + [word[index]])

        generate_splits(0, [])
        return ans