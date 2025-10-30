class Solution:
    def clearDigits(self, s: str) -> str:
        # Convert the string to a list for easier manipulation
        s_list = list(s)
        
        # Iterate over the list until all digits are removed
        while any(char.isdigit() for char in s_list):
            for i in range(len(s_list)):
                if s_list[i].isdigit():
                    # Find the closest non-digit character to the left
                    for j in range(i - 1, -1, -1):
                        if not s_list[j].isdigit():
                            # Remove the non-digit character
                            s_list.pop(j)
                            # Remove the digit
                            s_list.pop(i - 1)  # i-1 because we removed one element before
                            break
                    break
        
        # Join the list back into a string
        return ''.join(s_list)