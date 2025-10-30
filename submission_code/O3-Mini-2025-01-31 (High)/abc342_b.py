def main():
    import sys
    input_data = sys.stdin.read().split()
    # Parse the input.
    # First integer is N
    idx = 0
    N = int(input_data[idx])
    idx += 1
    
    # Next N integers are the people order.
    people = []
    for i in range(N):
        people.append(int(input_data[idx]))
        idx += 1
        
    # Build a dictionary mapping person number to their position (0-indexed).
    position = {}
    for pos, person in enumerate(people):
        position[person] = pos
        
    # Next integer is Q: number of queries.
    Q = int(input_data[idx])
    idx += 1
    
    # For each query, compare positions of persons A and B.
    output = []
    for _ in range(Q):
        A = int(input_data[idx])
        B = int(input_data[idx+1])
        idx += 2
        
        # The person with a lower position value is further to the front.
        if position[A] < position[B]:
            output.append(str(A))
        else:
            output.append(str(B))
    
    # Print each result on a new line.
    sys.stdout.write("
".join(output))
    
if __name__ == '__main__':
    main()