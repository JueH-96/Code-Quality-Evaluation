class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []
        
        def backtrack(current_string):
            if len(current_string) == n:
                valid = True
                for i in range(len(current_string) - 1):
                    substring = current_string[i:i+2]
                    if '1' not in substring:
                        valid = False
                        break
                if valid:
                    result.append(current_string)
                return

            for digit in ['0', '1']:
                backtrack(current_string + digit)

        backtrack("")
        return result