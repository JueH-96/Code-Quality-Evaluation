class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each hour value
        hour_count = Counter(hours)
        pairs = 0
        
        # Iterate through the unique hours
        for hour in hour_count:
            # If the hour is a multiple of 24, count the pairs within this group
            if hour % 24 == 0:
                pairs += hour_count[hour] * (hour_count[hour] - 1) // 2
            else:
                # Find the complementary hour that would sum up to a multiple of 24
                complement = 24 - (hour % 24)
                # If the complement exists in the list, count the pairs
                if complement in hour_count:
                    pairs += hour_count[hour] * hour_count[complement]
                # To avoid double counting, set the complement count to 0
                hour_count[complement] = 0
        
        return pairs