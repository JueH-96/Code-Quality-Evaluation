class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        count = 0
        n = len(hours)
        
        # Use a dictionary to count occurrences of each hour value
        hour_count = {}
        
        for hour in hours:
            # Calculate the complement that would form a complete day
            complement = (24 - (hour % 24)) % 24
            
            # If the complement exists in the dictionary, add its count to the result
            if complement in hour_count:
                count += hour_count[complement]
            
            # Update the count of the current hour in the dictionary
            if hour in hour_count:
                hour_count[hour] += 1
            else:
                hour_count[hour] = 1
        
        return count