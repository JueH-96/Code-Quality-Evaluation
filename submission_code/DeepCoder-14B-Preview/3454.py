class Solution:
    def clearDigits(self, s: str) -> str:
        while True:
            # Find the first digit
            d_index = -1
            for i in range(len(s)):
                if s[i].isdigit():
                    d_index = i
                    break
            if d_index == -1:
                break  # No more digits
            
            # Find the closest non-digit to the left of the digit
            n_index = -1
            for i in range(d_index - 1, -1, -1):
                if not s[i].isdigit():
                    n_index = i
                    break
            if n_index == -1:
                break  # No non-digit found (shouldn't happen as per problem statement)
            
            # Create the new string by excluding the two characters
            new_s = s[:n_index] + s[n_index+1:d_index] + s[d_index+1:]
            s = new_s
        
        return s