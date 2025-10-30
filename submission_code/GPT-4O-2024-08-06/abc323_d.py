# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    slimes = {}
    
    index = 1
    for _ in range(N):
        S = int(data[index])
        C = int(data[index + 1])
        index += 2
        if S in slimes:
            slimes[S] += C
        else:
            slimes[S] = C
    
    # Sort the sizes
    sizes = sorted(slimes.keys())
    
    for size in sizes:
        count = slimes[size]
        # Number of pairs we can form
        pairs = count // 2
        # Remaining slimes of this size
        remaining = count % 2
        
        # Update the current size with remaining slimes
        slimes[size] = remaining
        
        # Add the new slimes of size 2 * size
        if pairs > 0:
            new_size = 2 * size
            if new_size in slimes:
                slimes[new_size] += pairs
            else:
                slimes[new_size] = pairs
    
    # Calculate the total number of slimes left
    total_slimes = sum(slimes[size] for size in slimes)
    print(total_slimes)