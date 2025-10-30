class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Generate all possible strings by swapping characters at indices i and j where j - i = 2
        def generate_variations(s):
            variations = set()
            variations.add(s)
            variations.add(s[0] + s[3] + s[2] + s[1])  # Swap index 1 and 3
            variations.add(s[3] + s[1] + s[0] + s[2])  # Swap index 0 and 2
            variations.add(s[2] + s[1] + s[0] + s[3])  # Swap index 0 and 2, then swap index 1 and 3
            return variations

        # Generate variations for both s1 and s2
        variations_s1 = generate_variations(s1)
        variations_s2 = generate_variations(s2)

        # Check if there is any common variation between s1 and s2
        return not variations_s1.isdisjoint(variations_s2)