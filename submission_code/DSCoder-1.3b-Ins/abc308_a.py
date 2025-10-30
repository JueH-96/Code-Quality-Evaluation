# YOUR CODE HERE

# Read the inputs from stdin
inputs = list(map(int, input().split()))

# Check if the sequence is monotonically non-decreasing
if all(inputs[i] <= inputs[i + 1] for i in range(len(inputs) - 1)):
    # Check if all the numbers are between 100 and 675 inclusive and are multiples of 25
    if all(100 <= inputs[i] <= 675 and inputs[i] % 25 == 0 for i in range(len(inputs))):
        print("Yes")
    else:
        print("No")
else:
    print("No")