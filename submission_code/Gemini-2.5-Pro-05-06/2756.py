from typing import List

class Solution:
  def buyChoco(self, prices: List[int], money: int) -> int:
    # Sort the prices array in non-decreasing (ascending) order.
    # This allows us to easily find the two chocolates with the minimum combined price.
    prices.sort()
    
    # After sorting, the first element (prices[0]) is the price of the cheapest chocolate,
    # and the second element (prices[1]) is the price of the second cheapest chocolate.
    # The problem constraints state `2 <= prices.length <= 50`, 
    # so `prices[0]` and `prices[1]` are guaranteed to exist.
    
    min_price1 = prices[0]
    min_price2 = prices[1]
    
    # Calculate the sum of the prices of these two cheapest chocolates.
    cost_of_two_chocolates = min_price1 + min_price2
    
    # According to the problem, we must buy exactly two chocolates
    # and have non-negative leftover money.
    # This means `cost_of_two_chocolates` must be less than or equal to `money`.
    
    if cost_of_two_chocolates <= money:
      # If the cost is affordable, we buy these two chocolates.
      # The amount of money leftover is `money - cost_of_two_chocolates`.
      leftover_money = money - cost_of_two_chocolates
      return leftover_money
    else:
      # If the cost of the two cheapest chocolates is greater than `money`,
      # it's impossible to buy any two chocolates without going into debt
      # (since we chose the two cheapest ones).
      # In this scenario, the problem states we should return the initial `money`.
      return money