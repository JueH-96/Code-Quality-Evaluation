# YOUR CODE HERE
def check_conditions(sequence):
    # Check if sequence is monotonically non-decreasing
    if not all(sequence[i] <= sequence[i+1] for i in range(len(sequence)-1)):
        return False
    
    # Check if all numbers are between 100 and 675 inclusive and multiples of 25
    return all(100 <= num <= 675 and num % 25 == 0 for num in sequence)

# Read input
sequence = list(map(int, input().split()))

# Check conditions and print result
print("Yes" if check_conditions(sequence) else "No")