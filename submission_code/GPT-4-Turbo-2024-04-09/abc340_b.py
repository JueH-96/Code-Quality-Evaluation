import sys
input = sys.stdin.read

def process_queries():
    data = input().split()
    Q = int(data[0])
    queries = data[1:]
    
    A = []
    results = []
    
    index = 0
    for _ in range(Q):
        query_type = int(queries[index])
        number = int(queries[index + 1])
        
        if query_type == 1:
            A.append(number)
        elif query_type == 2:
            # Find the k-th value from the end
            k = number
            results.append(A[-k])
        
        index += 2
    
    for result in results:
        print(result)

process_queries()