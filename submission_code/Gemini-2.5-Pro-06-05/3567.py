class Solution:
    def convertDateToBinary(self, date: str) -> str:
        """
        Converts a date string from "yyyy-mm-dd" format to its binary representation.

        Args:
            date: A string representing a date in "yyyy-mm-dd" format.

        Returns:
            A string with the year, month, and day converted to their binary
            representations, joined by hyphens.
        """
        
        # Split the date string "yyyy-mm-dd" into its components.
        # For example, "2080-02-29" becomes ['2080', '02', '29'].
        parts = date.split('-')
        
        # Create a list of the binary representations for each component
        # using a list comprehension. For each part (year, month, day string):
        # 1. Convert it to an integer using int().
        # 2. Convert the integer to a binary string using bin().
        # 3. Remove the '0b' prefix from the binary string by slicing [2:].
        binary_components = [bin(int(p))[2:] for p in parts]
        
        # Join the binary components back together with a hyphen separator.
        # For example, ['100000100000', '10', '11101'] becomes
        # "100000100000-10-11101".
        return "-".join(binary_components)