def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    Q = int(input_data[0])
    
    # We'll track frequencies of each integer in a dictionary
    freq = {}
    # Also track how many distinct integers are currently in the bag
    distinct_count = 0
    
    idx = 1
    output = []
    
    for _ in range(Q):
        query_type = int(input_data[idx])
        idx += 1
        
        if query_type == 1:
            x = int(input_data[idx])
            idx += 1
            # Add x
            if x not in freq:
                freq[x] = 0
            freq[x] += 1
            # If this is the first ball of type x
            if freq[x] == 1:
                distinct_count += 1
        
        elif query_type == 2:
            x = int(input_data[idx])
            idx += 1
            # Remove one ball of type x
            freq[x] -= 1
            # If we removed and freq goes to 0, it's no longer distinct
            if freq[x] == 0:
                distinct_count -= 1
                
        else: # query_type == 3
            # Print the number of distinct integers in the bag
            output.append(str(distinct_count))
    
    print("
".join(output))

if __name__ == "__main__":
    main()