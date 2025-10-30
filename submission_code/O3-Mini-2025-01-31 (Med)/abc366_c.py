def main():
    import sys
    input_data = sys.stdin.read().split()
    Q = int(input_data[0])
    results = []
    # Using a dictionary to store counts of each integer.
    counts = {}
    idx = 1
    # Process each query.
    for _ in range(Q):
        query_type = input_data[idx]
        idx += 1
        if query_type == '1':
            # Add ball with integer x.
            x = int(input_data[idx])
            idx += 1
            if x in counts:
                counts[x] += 1
            else:
                counts[x] = 1
        elif query_type == '2':
            # Remove ball with integer x.
            x = int(input_data[idx])
            idx += 1
            counts[x] -= 1
            if counts[x] == 0:
                del counts[x]
        elif query_type == '3':
            # Print the number of distinct integers in the bag.
            results.append(str(len(counts)))
    
    # Output all results.
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()