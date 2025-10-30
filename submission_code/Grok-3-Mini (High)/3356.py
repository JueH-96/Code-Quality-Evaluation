from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def get_substrings(s):
            subs = set()
            for i in range(len(s)):
                for j in range(i, len(s)):
                    subs.add(s[i:j+1])
            return subs
        
        sub_sets_all = [get_substrings(s) for s in arr]
        n = len(arr)
        answer = []
        for i in range(n):
            union_other = set()
            for j in range(n):
                if j != i:
                    union_other |= sub_sets_all[j]
            my_subs = sub_sets_all[i]
            candidates = [sub for sub in my_subs if sub not in union_other]
            if not candidates:
                answer.append("")
            else:
                ans_str = min(candidates, key=lambda x: (len(x), x))
                answer.append(ans_str)
        return answer