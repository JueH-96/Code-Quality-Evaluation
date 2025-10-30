# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = data[2]

# Count the number of days Takahashi goes out for a meal (1) and attends a competitive programming event (2)
meal_days = S.count('1')
event_days = S.count('2')

# Calculate the number of logo T-shirts needed
logo_t_shirts_needed = max(0, event_days - M)

# Calculate the number of plain T-shirts needed
plain_t_shirts_needed = max(0, meal_days - M)

# The total number of T-shirts needed is the maximum of logo and plain T-shirts needed
total_t_shirts_needed = max(logo_t_shirts_needed, plain_t_shirts_needed)

print(total_t_shirts_needed)