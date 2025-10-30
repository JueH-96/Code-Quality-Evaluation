import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
P = int(data[2])
L = list(map(int, data[3:]))

# Count how many people already have hair length at least T
current_count = sum(1 for l in L if l >= T)

# If the condition is already satisfied
if current_count >= P:
    print(0)
else:
    # Calculate the number of days needed for each person to reach hair length T
    days_needed = [(T - l) for l in L if l < T]
    # Sort the days needed
    days_needed.sort()
    # Find the number of days after which P or more people will have hair length at least T
    days = days_needed[P - current_count - 1]
    print(days)