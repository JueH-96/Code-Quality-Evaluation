# Read the input
N = int(input())
D = list(map(int, input().split()))

# Function to check if a date has a repdigit
def has_repdigit(month, day):
    month_str = str(month)
    day_str = str(day)
    return all(digit == month_str[0] for digit in month_str) and all(digit == day_str[0] for digit in day_str)

# Count the number of days with repdigit dates
count = 0
for month in range(1, N+1):
    for day in range(1, D[month-1]+1):
        if has_repdigit(month, day):
            count += 1

# Print the answer
print(count)