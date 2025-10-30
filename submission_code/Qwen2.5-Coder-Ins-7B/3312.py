class Solution:
    def countKeyChanges(self, s: str) -> int:
        count = 0
        last_key = None
        for char in s:
            current_key = char.lower()
            if current_key != last_key:
                count += 1
                last_key = current_key
        return count