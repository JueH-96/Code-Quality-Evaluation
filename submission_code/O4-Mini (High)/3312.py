class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0
        changes = 0
        prev_key = s[0].lower()
        for ch in s[1:]:
            curr_key = ch.lower()
            if curr_key != prev_key:
                changes += 1
                prev_key = curr_key
        return changes