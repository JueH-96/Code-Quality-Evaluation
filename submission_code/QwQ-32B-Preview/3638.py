class Solution:
    def makeStringGood(self, s: str) -> int:
        from collections import Counter
        freq = Counter(s)
        
        # Get the frequencies of each character
        freq_values = list(freq.values())
        
        # Find the minimum and maximum frequencies
        min_freq = min(freq_values)
        max_freq = max(freq_values)
        
        # Function to calculate the cost for a given target frequency
        def calculate_cost(target):
            total_cost = 0
            for f in freq_values:
                if f < target:
                    # Need to insert characters
                    total_cost += (target - f)
                else:
                    # Need to delete or change characters
                    # Changing a character to the next letter costs 1 operation per step
                    # But changing to the next letter is only beneficial if it helps in reducing the frequency
                    # In this problem, it's more efficient to consider deletion for excess characters
                    total_cost += (f - target)
            return total_cost
        
        # Iterate over possible target frequencies from min_freq to max_freq
        # Also consider target frequencies beyond max_freq if needed
        min_cost = float('inf')
        for target in range(min_freq, max_freq + 1):
            cost = calculate_cost(target)
            if cost < min_cost:
                min_cost = cost
        
        # Additionally, check for target frequencies beyond the current max_freq
        # In case it's cheaper to insert more characters to a higher frequency
        for target in range(max_freq + 1, max_freq + 27):  # Arbitrary upper limit for search
            cost = calculate_cost(target)
            if cost < min_cost:
                min_cost = cost
            else:
                # If cost starts increasing, break early
                break
        
        return min_cost