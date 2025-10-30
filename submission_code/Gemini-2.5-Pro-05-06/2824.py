class Solution:
    def isFascinating(self, n: int) -> bool:
        # Calculate 2*n and 3*n
        val2 = 2 * n
        val3 = 3 * n

        # Concatenate n, 2*n, and 3*n as strings
        # The problem specifies the order of concatenation: n, then 2*n, then 3*n.
        concatenated_str = str(n) + str(val2) + str(val3)

        # The conditions "contains all digits from 1 to 9 exactly once" and
        # "does not contain any 0's" together mean that the concatenated string
        # must be a permutation of "123456789".
        # We can check this by sorting the characters of the concatenated string
        # and comparing it to the string "123456789".
        
        # This comparison automatically handles all requirements:
        # 1. Correct length: If concatenated_str does not have length 9,
        #    its sorted version cannot equal "123456789".
        # 2. No zeros: If '0' is present in concatenated_str, its sorted version
        #    will contain '0' and thus cannot equal "123456789".
        # 3. All digits 1-9 exactly once: If there are duplicate non-zero digits,
        #    or if any digit from 1-9 is missing (assuming length 9 and no zeros),
        #    the sorted string will not be "123456789".
        
        return "".join(sorted(concatenated_str)) == "123456789"