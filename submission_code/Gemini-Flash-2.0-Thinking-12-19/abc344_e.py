def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())
    queries = []
    for _ in range(q):
        query_line = list(map(int, input().split()))
        queries.append(query_line)
    
    next_element = {}
    previous_element = {}
    head = None
    tail = None
    
    if n > 0:
        head = a[0]
        tail = a[-1]
        for i in range(n):
            previous_element[a[i]] = None
            next_element[a[i]] = None
        for i in range(n - 1):
            next_element[a[i]] = a[i+1]
            previous_element[a[i+1]] = a[i]
        previous_element[a[0]] = None
        next_element[a[n-1]] = None
        
    current_sequence = list(a)
    
    for query in queries:
        query_type = query[0]
        if query_type == 1:
            x, y = query[1], query[2]
            s = next_element[x]
            next_element[x] = y
            previous_element[y] = x
            next_element[y] = s
            if s is not None:
                previous_element[s] = y
            else:
                tail = y
            current_sequence_updated = []
            current_element = head
            while current_element is not None:
                current_sequence_updated.append(current_element)
                current_element = next_element[current_element]
            current_sequence = current_sequence_updated
        elif query_type == 2:
            x = query[1]
            p = previous_element[x]
            s = next_element[x]
            if p is not None and s is not None:
                next_element[p] = s
                previous_element[s] = p
            elif p is None and s is not None:
                head = s
                previous_element[s] = None
            elif p is not None and s is None:
                tail = p
                next_element[p] = None
            elif p is None and s is None:
                head = None
                tail = None
                
            del next_element[x]
            del previous_element[x]
            current_sequence_updated = []
            current_element = head
            while current_element is not None:
                current_sequence_updated.append(current_element)
                current_element = next_element[current_element]
            current_sequence = current_sequence_updated
            
    result_sequence = []
    current_element = head
    while current_element is not None:
        result_sequence.append(current_element)
        current_element = next_element[current_element]
        
    print(*(result_sequence))

if __name__ == '__main__':
    solve()