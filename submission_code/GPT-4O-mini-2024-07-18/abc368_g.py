def process_queries():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    A = list(map(int, data[1].split()))
    B = list(map(int, data[2].split()))
    Q = int(data[3])
    
    results = []
    
    for i in range(4, 4 + Q):
        query = list(map(int, data[i].split()))
        
        if query[0] == 1:  # Type 1: Update A
            _, idx, x = query
            A[idx - 1] = x
            
        elif query[0] == 2:  # Type 2: Update B
            _, idx, x = query
            B[idx - 1] = x
            
        elif query[0] == 3:  # Type 3: Calculate maximum value
            _, l, r = query
            l -= 1  # Convert to 0-based index
            r -= 1
            
            v = 0
            for i in range(l, r + 1):
                # We can either add A[i] or multiply by B[i]
                add_option = v + A[i]
                multiply_option = v * B[i]
                v = max(add_option, multiply_option)
                
            results.append(v)
    
    # Print all results for type 3 queries
    sys.stdout.write('
'.join(map(str, results)) + '
')