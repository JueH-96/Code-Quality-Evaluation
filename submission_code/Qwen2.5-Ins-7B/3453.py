from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        if n == 1:
            return ["0", "1"]
        
        valid_strings = ["0", "1"]
        
        for _ in range(2, n + 1):
            new_valid_strings = set()
            for s in valid_strings:
                new_strings = ["0" + s, "1" + s]
                for ns in new_strings:
                    if all(ns[i:i+2].count("1") >= 1 for i in range(len(ns) - 1)):
                        new_valid_strings.add(ns)
            valid_strings = list(new_valid_strings)
        
        return valid_strings