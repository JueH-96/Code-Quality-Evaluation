from collections import defaultdict
from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        results = [""] * len(arr)
        str_to_idx = defaultdict(set)
        for i, str_i in enumerate(arr):
            str_to_idx[str_i].add(i)
        
        for i, str_i in enumerate(arr):
            for length in range(1, len(str_i) + 1):
                for start in range(len(str_i) - length + 1):
                    sub_str = str_i[start:start+length]
                    if len(str_to_idx[sub_str]) == 1 or (len(str_to_idx[sub_str]) == 2 and i in str_to_idx[sub_str] and "" in str_to_idx[sub_str]):
                        results[i] = sub_str
                        break
                else:
                    continue
                break
        
        return results