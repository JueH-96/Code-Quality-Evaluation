# YOUR CODE HERE
def process_queries(queries):
    from collections import defaultdict
    
    # This will store the count of each integer in the bag
    bag = defaultdict(int)
    
    # This will store the number of different integers in the bag
    unique_count = 0
    
    # This will store the results of type 3 queries
    results = []
    
    for query in queries:
        parts = query.split()
        if parts[0] == '1':
            # Add a ball with integer x
            x = int(parts[1])
            if bag[x] == 0:
                unique_count += 1
            bag[x] += 1
        elif parts[0] == '2':
            # Remove a ball with integer x
            x = int(parts[1])
            if bag[x] == 1:
                unique_count -= 1
            bag[x] -= 1
        elif parts[0] == '3':
            # Print the number of different integers
            results.append(unique_count)
    
    return results

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split('
')
    
    Q = int(data[0])
    queries = data[1:Q+1]
    
    results = process_queries(queries)
    
    for result in results:
        print(result)