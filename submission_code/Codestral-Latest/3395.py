class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        from math import gcd
        from functools import reduce

        # Count the frequency of each character in the string
        char_count = Counter(s)

        # Find the greatest common divisor (GCD) of the frequencies
        frequencies = list(char_count.values())
        gcd_value = reduce(gcd, frequencies)

        # The minimum length of the string t is the sum of the frequencies divided by the GCD
        min_length = sum(frequencies) // gcd_value

        return min_length