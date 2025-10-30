def count_power_representations(N):
    # Set to keep track of unique values
    unique_values = set()
    
    # Iterate through possible values of a
    max_a = int(N**0.5)
    for a in range(1, max_a + 1):
        power = a * a  # Start with a^2
        while power <= N:
            unique_values.add(power)
            power *= a  # Calculate the next power
    
    return len(unique_values)

# Read input
N = int(input())

# Calculate and print the answer
answer = count_power_representations(N)
print(answer)