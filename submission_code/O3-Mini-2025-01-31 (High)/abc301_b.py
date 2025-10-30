def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    sequence = list(map(int, data[1:1+N]))
    
    # Continue the process until every adjacent pair has a difference of 1
    while True:
        found_violation = False
        # Find the first position i where the adjacent difference is not 1
        for i in range(len(sequence) - 1):
            if abs(sequence[i] - sequence[i+1]) != 1:
                found_violation = True
                break
        # If all differences are 1, we're done
        if not found_violation:
            break
        
        # Insert the required numbers between sequence[i] and sequence[i+1]
        if sequence[i] < sequence[i+1]:
            # Ascending segment: insert numbers from sequence[i]+1 to sequence[i+1]-1
            inserted = list(range(sequence[i] + 1, sequence[i+1]))
        else:
            # Descending segment: insert numbers from sequence[i]-1 down to sequence[i+1]+1
            inserted = list(range(sequence[i] - 1, sequence[i+1], -1))
            
        # Reconstruct the sequence with the inserted numbers
        sequence = sequence[:i+1] + inserted + sequence[i+1:]
    
    # Print the final sequence with terms separated by spaces
    print(" ".join(map(str, sequence)))
    
if __name__ == '__main__':
    main()