# YOUR CODE HERE
def process_string_operations():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    S = list(data[1])  # Convert string to list for mutability
    Q = int(data[2])
    
    for i in range(3, 3 + Q):
        operation = data[i].split()
        t_i = int(operation[0])
        
        if t_i == 1:
            x_i = int(operation[1]) - 1  # Convert to 0-based index
            c_i = operation[2]
            S[x_i] = c_i
        elif t_i == 2:
            S = [c.lower() for c in S]
        elif t_i == 3:
            S = [c.upper() for c in S]
    
    print(''.join(S))

process_string_operations()