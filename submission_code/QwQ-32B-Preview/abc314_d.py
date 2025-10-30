def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    Q = int(data[2])
    operations = data[3:3+Q*3]
    
    # Initialize the string as a list for easy modification
    string_list = list(S)
    
    # Variable to track the last operation of type 2 or 3
    last_operation = None
    
    # Process each operation
    for i in range(Q):
        t = int(operations[i*3])
        x = int(operations[i*3+1])
        c = operations[i*3+2]
        
        if t == 1:
            # Change the x-th character to c
            string_list[x-1] = c
        elif t == 2:
            # Convert all uppercase letters to lowercase
            last_operation = 2
        elif t == 3:
            # Convert all lowercase letters to uppercase
            last_operation = 3
    
    # Apply the last operation of type 2 or 3 if any
    if last_operation == 2:
        # Convert all uppercase letters to lowercase
        for i in range(N):
            if string_list[i].isupper():
                string_list[i] = string_list[i].lower()
    elif last_operation == 3:
        # Convert all lowercase letters to uppercase
        for i in range(N):
            if string_list[i].islower():
                string_list[i] = string_list[i].upper()
    
    # Join the list into a string and print it
    print(''.join(string_list))

if __name__ == '__main__':
    main()