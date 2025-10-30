def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    Q = int(input_data[0])
    A = []
    result = []
    
    index = 1
    for _ in range(Q):
        query_type = input_data[index]
        if query_type == "1":
            # Append query
            x = int(input_data[index+1])
            A.append(x)
            index += 2
        elif query_type == "2":
            # Retrieve query
            k = int(input_data[index+1])
            # k-th from the end is at index -k
            result.append(str(A[-k]))
            index += 2
            
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()