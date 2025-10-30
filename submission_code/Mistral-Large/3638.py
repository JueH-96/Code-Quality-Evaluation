class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        import statistics

        # Step 1: Count the frequency of each character
        freq = Counter(s)

        # Step 2: Determine the target frequency
        frequencies = list(freq.values())
        target_freq = int(statistics.median(frequencies))

        # Step 3: Calculate the number of operations
        operations = 0
        for char, count in freq.items():
            if count < target_freq:
                operations += target_freq - count
            elif count > target_freq:
                operations += count - target_freq

        return operations