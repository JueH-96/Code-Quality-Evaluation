def main():
    import sys
    from collections import Counter
    
    # Read input data
    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    # Convert the rest to integers representing each person's number
    a = list(map(int, data[1:]))

    # Count the occurrences of each integer
    count = Counter(a)
    
    # Initialize the candidate variable: label = -1 and max_val = -1 as indicator
    result_label = -1
    max_val = -1
    
    # Iterate over each person (using 0-indexing, convert to 1-indexed label)
    for i, val in enumerate(a):
        if count[val] == 1 and val > max_val:
            max_val = val
            result_label = i + 1
            
    print(result_label)

# Calling main to execute the solution when the script runs
if __name__ == '__main__':
    main()