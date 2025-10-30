class Solution:
    def convertDateToBinary(self, date: str) -> str:
        """
        Converts a date string in yyyy-mm-dd format to its binary representation.

        Args:
            date: The date string in yyyy-mm-dd format.

        Returns:
            The binary representation of the date string.
        """
        # Split the date string into year, month, and day components
        year_str, month_str, day_str = date.split('-')

        # Convert the components from strings to integers
        year = int(year_str)
        month = int(month_str)
        day = int(day_str)

        # Convert each integer to its binary representation
        # Use bin() and slice [2:] to remove the '0b' prefix
        year_bin = bin(year)[2:]
        month_bin = bin(month)[2:]
        day_bin = bin(day)[2:]

        # Join the binary representations with hyphens
        binary_date = f"{year_bin}-{month_bin}-{day_bin}"

        return binary_date