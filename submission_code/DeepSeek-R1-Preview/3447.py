class Solution:
    def clearDigits(self, s: str) -> str:
        current_list = list(s)
        while True:
            # Find the first digit
            i = next((idx for idx, c in enumerate(current_list) if c.isdigit()), None)
            if i is None:
                break
            # Find the closest non-digit to the left
            j = None
            for idx in range(i - 1, -1, -1):
                if not current_list[idx].isdigit():
                    j = idx
                    break
            # Create the new list without the two characters
            new_list = current_list[:j] + current_list[j+1:i] + current_list[i+1:]
            current_list = new_list
        return ''.join(current_list)