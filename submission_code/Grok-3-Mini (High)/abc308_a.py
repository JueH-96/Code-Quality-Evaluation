# Read input from stdin and split into a list of strings
data = input().split()
# Convert the list of strings to a list of integers
S = [int(x) for x in data]
# Check if all elements are within [100, 675] and multiples of 25
range_multiple_ok = all(100 <= num <= 675 and num % 25 == 0 for num in S)
# Check if the sequence is monotonically non-decreasing
order_ok = all(S[i] <= S[i+1] for i in range(7))
# Print Yes if both conditions are satisfied, otherwise No
if range_multiple_ok and order_ok:
    print("Yes")
else:
    print("No")