class Solution:
    def validStrings(self, n: int) -> List[str]:
        result = []

        def generate(length, current_string):
            if length == n:
                result.append(current_string)
                return

            # Try appending '1'
            generate(length + 1, current_string + '1')

            # Try appending '0' only if the last char is not '0'
            if not current_string or current_string[-1] != '0':
                generate(length + 1, current_string + '0')

        if n == 0:
            return []

        if n == 1:
            return ["0", "1"]

        generate(0, "")

        # Filter out strings containing "00"
        valid_result = [s for s in result if "00" not in s]
        return valid_result