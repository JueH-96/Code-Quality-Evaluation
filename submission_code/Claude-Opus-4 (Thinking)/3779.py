class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        pizzas.sort(reverse=True)
        n = len(pizzas)
        days = n // 4
        odd_days = (days + 1) // 2
        even_days = days - odd_days
        
        # For odd days: pair each of the largest pizzas with 3 smallest pizzas
        # This ensures the max (Z) is as large as possible
        total = sum(pizzas[:odd_days])
        
        # Remaining pizzas after using top odd_days and bottom 3*odd_days
        remaining = pizzas[odd_days:n - 3 * odd_days]
        remaining.sort()
        
        # For even days: form groups from remaining pizzas
        # Take the 3rd element (Y) from each group of 4
        for i in range(even_days):
            total += remaining[i * 4 + 2]
        
        return total