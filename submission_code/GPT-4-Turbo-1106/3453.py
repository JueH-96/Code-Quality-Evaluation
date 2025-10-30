class Solution:
    def validStrings(self, n: int) -> List[str]:
        # Base case: if n is 1, the valid strings are "0" and "1"
        if n == 1:
            return ["0", "1"]
        
        # Recursive case: build the valid strings of length n based on those of length n-1
        def buildValidStrings(n):
            if n == 1:
                return ["0", "1"]
            else:
                # Get all valid strings of length n-1
                prev_strings = buildValidStrings(n-1)
                valid_strings = []
                # Append "1" to all strings from the previous step
                valid_strings.extend([s + "1" for s in prev_strings])
                # Append "0" only to strings that end with "1"
                valid_strings.extend([s + "0" for s in prev_strings if s[-1] == "1"])
                return valid_strings
        
        # Generate and return all valid strings of length n
        return buildValidStrings(n)