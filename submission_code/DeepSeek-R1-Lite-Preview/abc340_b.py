def main():
    import sys
    input_lines = sys.stdin.read().splitlines()
    Q = int(input_lines[0])
    queries = input_lines[1:1+Q]
    
    A = []  # Sequence A
    results = []  # To store results of type 2 queries
    
    for query in queries:
        parts = query.split()
        if parts[0] == '1':
            x = int(parts[1])
            A.append(x)
        elif parts[0] == '2':
            k = int(parts[1])
            results.append(str(A[-k]))
    
    print('
'.join(results))

if __name__ == '__main__':
    main()