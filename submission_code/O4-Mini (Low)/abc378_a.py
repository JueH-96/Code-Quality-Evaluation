def main():
    import sys
    # Read four integers from standard input
    colors = list(map(int, sys.stdin.read().split()))
    
    # Count occurrences of each color
    counts = [0] * 5  # indices 1 to 4
    for c in colors:
        counts[c] += 1
    
    # For each color, we can discard floor(count/2) pairs
    operations = sum(cnt // 2 for cnt in counts)
    
    # Print the result
    print(operations)

if __name__ == "__main__":
    main()