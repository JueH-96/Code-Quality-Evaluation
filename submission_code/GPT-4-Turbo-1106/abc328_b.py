def is_repdigit(num):
    return str(num) == str(num)[0] * len(str(num))

def count_repdigit_dates(N, days_in_months):
    count = 0
    for month in range(1, N + 1):
        for day in range(1, days_in_months[month - 1] + 1):
            if is_repdigit(month) and is_repdigit(day):
                count += 1
    return count

# Read input
N = int(input().strip())
days_in_months = list(map(int, input().strip().split()))

# Calculate and print the answer
print(count_repdigit_dates(N, days_in_months))