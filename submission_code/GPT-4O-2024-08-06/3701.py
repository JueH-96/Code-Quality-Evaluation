class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""  # Impossible to form a good caption with less than 3 characters

        # Helper function to find the lexicographically smallest good caption
        def make_good(start, end, char):
            return char * (end - start + 1)

        result = []
        i = 0

        while i < n:
            j = i
            # Find the length of the current group of the same character
            while j < n and caption[j] == caption[i]:
                j += 1

            length = j - i

            if length >= 3:
                # If the group is already good, add it to the result
                result.append(caption[i] * length)
            else:
                # If the group is not good, try to extend it
                if i > 0 and j < n:
                    # Check if we can extend by changing the current group to the previous or next character
                    if caption[i - 1] == caption[j]:
                        # We can extend by changing the current group to the previous or next character
                        result.append(make_good(i, j - 1, caption[i - 1]))
                    else:
                        # If we can't extend, it's impossible to make a good caption
                        return ""
                else:
                    # If at the start or end, it's impossible to make a good caption
                    return ""

            i = j

        return ''.join(result)