class Solution:
    def validStrings(self, n: int) -> List[str]:
        ans = []
        def backtrack(position, str_):
            if position == n:
                ans.append(str_)
                return
            
            if position == 0 or str_[position - 1] == '1':
                backtrack(position + 1, str_ + '0')
                backtrack(position + 1, str_ + '1')
            else:
                backtrack(position + 1, str_ + '1')
                
        backtrack(0, '')
        return ans