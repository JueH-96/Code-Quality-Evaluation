# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
B = int(data[2])
D = list(map(int, data[3:]))

# Check if all plans can be scheduled on holidays
def can_all_be_holidays(N, A, B, D):
    # The last day of the week is A + B
    last_day = A + B
    # Check if the difference between consecutive days is a multiple of the week length
    for i in range(1, N):
        if (D[i] - D[i-1]) % last_day != 0:
            return "No"
    # Check if the first day of the week is a holiday
    if D[0] % last_day <= A:
        return "Yes"
    else:
        return "No"

# Print the result
print(can_all_be_holidays(N, A, B, D))