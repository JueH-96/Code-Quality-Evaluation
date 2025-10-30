from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        # Initialize a counter for seniors
        senior_count = 0
        
        # Iterate over each passenger's details
        for detail in details:
            # Extract the age from the details string
            age = int(detail[11:13])
            
            # Check if the passenger is a senior (age > 60)
            if age > 60:
                # Increment the senior count
                senior_count += 1
        
        # Return the total count of seniors
        return senior_count