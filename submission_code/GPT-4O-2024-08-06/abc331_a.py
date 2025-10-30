# YOUR CODE HERE
def next_day(M, D, y, m, d):
    # Check if it's the last day of the month
    if d == D:
        d = 1
        # Check if it's the last month of the year
        if m == M:
            m = 1
            y += 1
        else:
            m += 1
    else:
        d += 1
    
    return y, m, d

# Reading input
import sys
input = sys.stdin.read
data = input().strip().split()

M = int(data[0])
D = int(data[1])
y = int(data[2])
m = int(data[3])
d = int(data[4])

# Calculate the next day
y_next, m_next, d_next = next_day(M, D, y, m, d)

# Print the result
print(y_next, m_next, d_next)