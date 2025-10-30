def find_sequence(n, ranges):
    # Initialize the sequence with the left bounds
    sequence = [L for L, R in ranges]
    total_sum = sum(sequence)
    
    # Adjust the sequence to sum up to zero
    for i in range(n):
        L, R = ranges[i]
        # Calculate the maximum value we can add to X_i without exceeding R_i
        max_add = R - sequence[i]
        # Calculate the required value to add to make the sum zero
        required_add = min(-total_sum, max_add)
        # Adjust X_i and the total sum
        sequence[i] += required_add
        total_sum += required_add
        # If the sum is already zero, we are done
        if total_sum == 0:
            return sequence
    return None

# Read input
N = int(input().strip())
ranges = [tuple(map(int, input().split())) for _ in range(N)]

# Find a valid sequence or determine it's impossible
sequence = find_sequence(N, ranges)

# Output the result
if sequence:
    print("Yes")
    print(' '.join(map(str, sequence)))
else:
    print("No")