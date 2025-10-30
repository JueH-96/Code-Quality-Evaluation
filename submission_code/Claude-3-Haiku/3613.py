from collections import defaultdict

class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:
        # Create dictionaries to store the conversion rates for each day
        day1_rates = defaultdict(dict)
        day2_rates = defaultdict(dict)
        
        # Populate the conversion rate dictionaries
        for i in range(len(pairs1)):
            day1_rates[pairs1[i][0]][pairs1[i][1]] = rates1[i]
        for i in range(len(pairs2)):
            day2_rates[pairs2[i][0]][pairs2[i][1]] = rates2[i]
        
        # Initialize the current amount with the initial currency
        current_amount = 1.0
        current_currency = initialCurrency
        
        # Perform conversions on day 1
        for i in range(len(pairs1)):
            if current_currency == pairs1[i][0]:
                current_amount *= day1_rates[pairs1[i][0]][pairs1[i][1]]
                current_currency = pairs1[i][1]
        
        # Perform conversions on day 2
        for i in range(len(pairs2)):
            if current_currency == pairs2[i][0]:
                current_amount *= day2_rates[pairs2[i][0]][pairs2[i][1]]
                current_currency = pairs2[i][1]
            elif current_currency == pairs2[i][1]:
                current_amount /= day2_rates[pairs2[i][0]][pairs2[i][1]]
                current_currency = pairs2[i][0]
        
        # Return the maximum amount of the initial currency
        return round(current_amount, 5)