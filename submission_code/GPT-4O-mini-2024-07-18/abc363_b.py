# YOUR CODE HERE
def days_until_hair_length_reaches_threshold(N, T, P, lengths):
    # Check how many people currently have hair length >= T
    current_count = sum(1 for length in lengths if length >= T)
    
    # If the current count is already >= P, return 0
    if current_count >= P:
        return 0
    
    # Calculate the number of days required
    days = 0
    while True:
        days += 1
        # Count how many people will have hair length >= T after 'days' days
        new_count = sum(1 for length in lengths if length + days >= T)
        
        if new_count >= P:
            return days

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
P = int(data[2])
lengths = list(map(int, data[3:]))

# Get the result and print it
result = days_until_hair_length_reaches_threshold(N, T, P, lengths)
print(result)