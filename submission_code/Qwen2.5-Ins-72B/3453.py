from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        def generate_strings(length: int, prev: str, result: List[str]):
            if length == 0:
                result.append(prev)
                return
            if length == 1 or prev[-1] == '1':
                generate_strings(length - 1, prev + '0', result)
            if length >= 1:
                generate_strings(length - 1, prev + '1', result)
        
        result = []
        if n == 1:
            return ["0", "1"]
        generate_strings(n, "", result)
        return result