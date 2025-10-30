import sys
data = sys.stdin.read().strip()
N, M = map(int, data.split())
sum_x = 1  # Initialize sum with N^0
exceeded = False
power_val = 1  # Start with N^0
for i in range(1, M + 1):  # Loop from 1 to M to add N^i
    power_val *= N  # Update power_val to N^i
    sum_x += power_val  # Add the current term to sum
    if sum_x > 1000000000:  # Check if sum exceeds 10^9
        exceeded = True
        break  # Stop further computation
if exceeded:
    print("inf")  # Print "inf" if sum exceeded 10^9
else:
    print(sum_x)  # Print the sum if it did not exceed 10^9