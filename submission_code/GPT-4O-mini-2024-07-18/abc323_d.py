def min_slimes_after_synthesis(N, sizes_counts):
    from collections import defaultdict
    
    # Dictionary to hold the count of slimes by size
    slime_count = defaultdict(int)
    
    # Fill the dictionary with the initial sizes and counts
    for size, count in sizes_counts:
        slime_count[size] += count
    
    # To keep track of the number of slimes after synthesis
    total_slimes = sum(slime_count.values())
    
    # Process the sizes in sorted order
    for size in sorted(slime_count.keys()):
        count = slime_count[size]
        
        # If we have at least two slimes of this size, we can synthesize
        if count >= 2:
            # Number of new slimes created
            new_slimes = count // 2
            # Remaining slimes of this size after synthesis
            remaining_slimes = count % 2
            
            # Update the total slimes
            total_slimes -= new_slimes  # We lose new_slimes because we synthesize them
            
            # Add the new size slime
            new_size = size * 2
            slime_count[new_size] += new_slimes
            
            # If there are remaining slimes of the current size, we need to account for them
            if remaining_slimes > 0:
                slime_count[size] = 1
            else:
                del slime_count[size]  # Remove the size if no slimes are left
            
            # After synthesis, we need to check the new size slimes
            if new_slimes > 0:
                total_slimes += new_slimes  # Add the new slimes created
            
    return total_slimes

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
sizes_counts = [tuple(map(int, line.split())) for line in data[1:N+1]]

# Get the result
result = min_slimes_after_synthesis(N, sizes_counts)

# Print the result
print(result)