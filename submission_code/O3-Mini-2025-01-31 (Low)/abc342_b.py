def main():
    import sys
    input_data = sys.stdin.read().split()
    # Parse input
    index = 0
    N = int(input_data[index])
    index += 1
    # P contains the person number in order from front to back
    P = list(map(int, input_data[index:index+N]))
    index += N
    Q = int(input_data[index])
    index += 1
    
    # Precompute a dictionary from person number to their position (0-indexed)
    position = {person: i for i, person in enumerate(P)}
    
    results = []
    for _ in range(Q):
        A = int(input_data[index])
        B = int(input_data[index+1])
        index += 2
        
        # Compare their positions (lower index means more front)
        if position[A] < position[B]:
            results.append(str(A))
        else:
            results.append(str(B))
    
    # Output the results line by line
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()