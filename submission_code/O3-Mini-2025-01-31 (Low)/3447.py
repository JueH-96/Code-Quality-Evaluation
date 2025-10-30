class Solution:
    def clearDigits(self, s: str) -> str:
        # Convert the string to a list for mutable operations.
        lst = list(s)
        # Continue until no digit exists in the list.
        while any(ch.isdigit() for ch in lst):
            # Find the first digit's index.
            for i, ch in enumerate(lst):
                if ch.isdigit():
                    digit_index = i
                    break
            # Now find the closest non-digit character to the left of this digit.
            # It is guaranteed by the problem that one exists.
            for j in range(digit_index - 1, -1, -1):
                if not lst[j].isdigit():
                    left_letter_index = j
                    break
            # Remove the found digit and the associated left letter.
            # Since the digit is right to the letter, remove the digit first to keep indices valid.
            del lst[digit_index]
            del lst[left_letter_index]
        return "".join(lst)
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.clearDigits("abc"))  # Output should be "abc"
    print(sol.clearDigits("cb34")) # Output should be ""