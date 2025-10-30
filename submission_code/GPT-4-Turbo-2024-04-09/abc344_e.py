import sys
input = sys.stdin.read

def process_queries():
    data = input().split()
    index = 0
    
    # Read the initial sequence length N
    N = int(data[index])
    index += 1
    
    # Read the initial sequence A
    A = list(map(int, data[index:index + N]))
    index += N
    
    # Read the number of queries Q
    Q = int(data[index])
    index += 1
    
    # Dictionary to quickly find the index of each element in A
    position = {A[i]: i for i in range(N)}
    
    # Process each query
    for _ in range(Q):
        query = data[index]
        if query == '1':
            # Insert y immediately after x
            x = int(data[index + 1])
            y = int(data[index + 2])
            pos_x = position[x]
            A.insert(pos_x + 1, y)
            # Update positions of elements after inserted position
            for i in range(pos_x + 1, len(A)):
                position[A[i]] = i
            index += 3
        elif query == '2':
            # Remove element x
            x = int(data[index + 1])
            pos_x = position[x]
            A.pop(pos_x)
            # Update positions of elements after removed position
            for i in range(pos_x, len(A)):
                position[A[i]] = i
            index += 2
    
    # Print the final sequence
    print(' '.join(map(str, A)))

process_queries()