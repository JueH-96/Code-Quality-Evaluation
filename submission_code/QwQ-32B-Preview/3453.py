from typing import List

class Solution:
    def validStrings(self, n: int) -> List[str]:
        current_list = [""]
        
        for _ in range(n):
            new_list = []
            for s in current_list:
                if not s or s[-1] == "1":
                    new_list.append(s + "0")
                    new_list.append(s + "1")
                elif s[-1] == "0":
                    new_list.append(s + "1")
            current_list = new_list
        return current_list