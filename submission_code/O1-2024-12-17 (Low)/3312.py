class Solution:
    def countKeyChanges(self, s: str) -> int:
        if not s:
            return 0

        # We'll track the number of key changes based on the letter ignoring its case.
        changes = 0
        prev_letter = s[0].lower()

        for i in range(1, len(s)):
            current_letter = s[i].lower()
            if current_letter != prev_letter:
                changes += 1
            prev_letter = current_letter

        return changes