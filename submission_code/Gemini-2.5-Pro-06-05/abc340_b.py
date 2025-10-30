import sys

def solve():
    """
    Processes a series of queries on a sequence.
    """
    # Initialize an empty sequence (list)
    A = []
    
    # Read the number of queries
    try:
        Q = int(sys.stdin.readline())
    except (IOError, ValueError):
        # Handle case where input is empty or not an integer
        return
        
    # Process each query
    for _ in range(Q):
        line = sys.stdin.readline().strip()
        if not line:
            continue
            
        parts = line.split()
        query_type = int(parts[0])
        
        if query_type == 1:
            # Type 1: Append x to the end of A
            x = int(parts[1])
            A.append(x)
        elif query_type == 2:
            # Type 2: Find the k-th value from the end of A
            k = int(parts[1])
            # Python's negative indexing makes this straightforward.
            # A[-1] is the last element, A[-k] is the k-th from the end.
            print(A[-k])

# Run the solution
solve()