class Solution:
    def finalString(self, s: str) -> str:
        res = []                    # characters currently on the screen
        
        for ch in s:
            if ch == 'i':
                # faulty 'i' key: reverse the current text
                res.reverse()
            else:
                # any other character is appended normally
                res.append(ch)
                
        return ''.join(res)