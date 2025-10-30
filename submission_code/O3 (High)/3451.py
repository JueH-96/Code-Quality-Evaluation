class Solution:
    def compressedString(self, word: str) -> str:
        # result pieces will be collected in a list for efficiency
        res = []
        
        # Initialize the first run
        prev = word[0]
        cnt = 0
        
        for ch in word:
            if ch == prev and cnt < 9:      # continue the current run (max 9)
                cnt += 1
            else:                           # end of a run (char changed or cnt reached 9)
                if cnt:                     # append completed run
                    res.append(str(cnt))
                    res.append(prev)
                prev = ch                   # start new run
                cnt = 1
        
        # append the final run
        res.append(str(cnt))
        res.append(prev)
        
        # join list into the final compressed string
        return ''.join(res)