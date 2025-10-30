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
        slimes[S] = C
        index += 2
    
    # Dictionary to hold the counts of slimes by size
    counts = {}
    
    # Read the sizes and counts
    for size, count in slimes.items():
        counts[size] = count
    
    # Sort the sizes
    sizes = sorted(counts.keys())
    
    # To hold the leftover slimes that cannot be synthesized further
    leftovers = 0
    
    # Process each size
    for size in sizes:
        if size in counts:
            # Get the number of slimes of this size
            count = counts[size]
            
            # Calculate how many pairs we can make and the leftover
            pairs = count // 2
            leftover = count % 2
            
            # If there's a leftover, it will be carried forward
            leftovers += leftover
            
            # If we can make pairs, the resulting size is 2 * size
            new_size = 2 * size
            new_count = pairs
            
            # Add the new slimes to the counts dictionary
            if new_size in counts:
                counts[new_size] += new_count
            else:
                counts[new_size] = new_count
    
    # The result is the sum of all leftovers
    print(leftovers)

if __name__ == "__main__":
    main()