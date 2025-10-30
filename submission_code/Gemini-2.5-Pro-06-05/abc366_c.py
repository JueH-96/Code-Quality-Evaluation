import sys

def solve():
    """
    Reads queries from standard input, processes them to simulate a bag of balls,
    and prints the number of distinct ball types for type 3 queries.
    """
    # A dictionary to store the counts of each ball's number.
    # Key: ball number (int), Value: count of balls with that number (int)
    bag_counts = {}

    # Read the total number of queries.
    try:
        num_queries = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # This path should not be taken given the problem constraints,
        # but it provides robustness for empty or invalid input.
        return

    # Process each query in order.
    for _ in range(num_queries):
        # Read a single query and split it into parts.
        query = sys.stdin.readline().split()
        
        query_type = int(query[0])
        
        if query_type == 1:
            # Type 1: Add a ball with integer x to the bag.
            x = int(query[1])
            # Increment the count for x. If x is new, it's initialized to 1.
            bag_counts[x] = bag_counts.get(x, 0) + 1
            
        elif query_type == 2:
            # Type 2: Remove one ball with integer x from the bag.
            x = int(query[1])
            # The problem guarantees that a ball with x exists.
            bag_counts[x] -= 1
            # If the count drops to zero, remove the integer type from the bag.
            if bag_counts[x] == 0:
                del bag_counts[x]
                
        elif query_type == 3:
            # Type 3: Print the number of different integers in the bag.
            # This is equal to the number of keys in the dictionary.
            print(len(bag_counts))

# Execute the solution.
solve()