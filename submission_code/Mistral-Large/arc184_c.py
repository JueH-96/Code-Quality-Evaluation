import sys

def count_mountain_folds(N, A):
    # Initialize the list to store the number of mountain folds
    mountain_folds = [0] * (2**100)

    # Define the function to determine the type of fold
    def fold_type(k):
        return (k & 1) == 0

    # Calculate the number of mountain folds for each position
    for i in range(1, 2**100 - A[-1]):
        count = 0
        for a in A:
            if i + a < 2**100 and fold_type(i + a):
                count += 1
        mountain_folds[i] = count

    # Find the maximum value in the mountain_folds list
    max_mountain_folds = max(mountain_folds[1:2**100 - A[-1]])
    return max_mountain_folds

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

# Calculate and print the result
result = count_mountain_folds(N, A)
print(result)