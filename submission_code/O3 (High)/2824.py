class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n as strings
        concatenated = f"{n}{2*n}{3*n}"
        
        # A fascinating number must produce exactly 9 digits,
        # each of '1'..'9' appearing once and no '0' present.
        if len(concatenated) != 9 or '0' in concatenated:
            return False
        
        # Sort the digits and compare with the perfect sequence '123456789'
        return ''.join(sorted(concatenated)) == '123456789'