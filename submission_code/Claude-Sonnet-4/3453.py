class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current_string):
            # Base case: if we've built a string of length n
            if len(current_string) == n:
                result.append(current_string)
                return
            
            # If this is the first character, we can add either '0' or '1'
            if len(current_string) == 0:
                backtrack(current_string + '0')
                backtrack(current_string + '1')
            else:
                # If the last character is '0', we can only add '1'
                if current_string[-1] == '0':
                    backtrack(current_string + '1')
                # If the last character is '1', we can add either '0' or '1'
                else:
                    backtrack(current_string + '0')
                    backtrack(current_string + '1')
        
        backtrack("")
        return result