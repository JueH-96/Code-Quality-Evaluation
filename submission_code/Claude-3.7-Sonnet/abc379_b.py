# YOUR CODE HERE
def max_strawberries(n, k, s):
    teeth = list(s)  # Convert string to list for easier modification
    strawberries = 0
    
    for i in range(n - k + 1):
        # Check if there's a sequence of K consecutive 'O's starting at index i
        if all(teeth[i+j] == 'O' for j in range(k)):
            # Eat a strawberry using these K teeth
            for j in range(k):
                teeth[i+j] = 'X'  # These teeth now have cavities
            strawberries += 1
    
    return strawberries

# Read input
n, k = map(int, input().strip().split())
s = input().strip()

# Print result
print(max_strawberries(n, k, s))