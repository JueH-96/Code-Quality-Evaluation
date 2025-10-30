# N M
# S
N, M = map(int, input().split())
S = input()

# Count the number of days he plans to go out for a meal and attend a competitive programming event
days_meal = S.count('1')
days_event = S.count('2')

# The number of T-shirts he needs to buy is the maximum of the days he plans to go out for a meal and attend a competitive programming event minus the number of plain T-shirts he already has
tshirts_to_buy = max(days_meal, days_event) - M

# Ensure the number of T-shirts to buy is non-negative
tshirts_to_buy = max(0, tshirts_to_buy)

print(tshirts_to_buy)