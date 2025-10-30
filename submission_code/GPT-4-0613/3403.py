class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        from collections import Counter
        counter = Counter(s)
        seen = set()
        stack = []
        last_occurrence = {c: i for i, c in enumerate(s)}
        
        for i, c in enumerate(s):
            counter[c] -= 1
            if c not in seen:
                while stack and c < stack[-1] and i < last_occurrence[stack[-1]]:
                    seen.remove(stack.pop())
                seen.add(c)
                stack.append(c)
                
        return len(stack)