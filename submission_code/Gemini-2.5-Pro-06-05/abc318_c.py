import sys

# Read N (number of days), D (passes per batch), and P (price per batch)
N, D, P = map(int, sys.stdin.readline().split())
# Read the list of daily fares
F = list(map(int, sys.stdin.readline().split()))

# To maximize the benefit of passes, we should use them on the most expensive days.
# Sort the fares in descending order.
F.sort(reverse=True)

# Initialize the total cost and a pointer for iterating through the fares.
total_cost = 0
i = 0

# Iterate through the sorted fares in chunks of size D.
while i < N:
    # Calculate the sum of fares for the next D days (or fewer if at the end of the trip).
    # Slicing `F[i:i+D]` handles the end case where fewer than D days remain.
    chunk_sum = sum(F[i : i + D])
    
    # Decide whether to buy a batch of passes or pay regular fares for this chunk.
    # It's better to buy a pass batch if its price P is less than the sum of fares.
    if P < chunk_sum:
        total_cost += P
        # Move to the start of the next chunk.
        i += D
    else:
        # If it's cheaper to pay fares for this chunk, it will be for all subsequent
        # (and cheaper) chunks as well, because the fares are sorted.
        # So, we pay the fares for all remaining days and stop.
        total_cost += sum(F[i:])
        break
        
# Print the final calculated minimum cost.
print(total_cost)