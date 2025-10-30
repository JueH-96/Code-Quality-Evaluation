class Solution:
    def clearDigits(self, s: str) -> str:
        arr = list(s)
        while True:
            idx_digit = None
            for i, char in enumerate(arr):
                if char.isdigit():
                    idx_digit = i
                    break
            if idx_digit is None:
                break
            
            idx_non_digit = idx_digit - 1
            while idx_non_digit >= 0 and arr[idx_non_digit].isdigit():
                idx_non_digit -= 1
            
            del arr[idx_digit]
            del arr[idx_non_digit]
        
        return ''.join(arr)