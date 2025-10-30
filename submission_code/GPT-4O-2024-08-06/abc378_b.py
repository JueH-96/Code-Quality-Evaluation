# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    
    collection_rules = []
    for _ in range(N):
        q_i = int(data[index])
        r_i = int(data[index + 1])
        collection_rules.append((q_i, r_i))
        index += 2
    
    Q = int(data[index])
    index += 1
    
    queries = []
    for _ in range(Q):
        t_j = int(data[index])
        d_j = int(data[index + 1])
        queries.append((t_j, d_j))
        index += 2
    
    results = []
    for t_j, d_j in queries:
        q_t, r_t = collection_rules[t_j - 1]
        
        if d_j % q_t == r_t:
            results.append(d_j)
        else:
            next_collection_day = d_j + (r_t - (d_j % q_t) + q_t) % q_t
            results.append(next_collection_day)
    
    for result in results:
        print(result)