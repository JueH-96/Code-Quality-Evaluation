# Read the input values
N, A, B = map(int, input().split())
D = list(map(int, input().split()))

# Check if all plans can be scheduled on holidays
def can_schedule_on_holidays(N, A, B, D):
    for d in D:
        if d % (A + B) > A or d % (A + B) == 0:
            return "No"
    return "Yes"

# Output the result
print(can_schedule_on_holidays(N, A, B, D))