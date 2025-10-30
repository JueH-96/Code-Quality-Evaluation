class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        result = []
        
        def backtrack(current_string):
            if len(current_string) == n:
                result.append(current_string)
                return
            
            for digit in ["0", "1"]:
                if len(current_string) > 0:
                    if len(current_string) >= 1 and current_string[-1] == "0" and digit == "0":
                        continue
                backtrack(current_string + digit)
        
        backtrack("")
        return result