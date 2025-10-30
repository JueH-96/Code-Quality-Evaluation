# YOUR CODE HERE
def min_ng_list_strings():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    product_names = data[1:]
    
    from collections import defaultdict
    
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    
    for name in product_names:
        start, end = name[0], name[1]
        out_degree[start] += 1
        in_degree[end] += 1
    
    start_mismatch = 0
    end_mismatch = 0
    
    for node in set(in_degree.keys()).union(set(out_degree.keys())):
        out_d = out_degree[node]
        in_d = in_degree[node]
        if out_d > in_d:
            start_mismatch += (out_d - in_d)
        elif in_d > out_d:
            end_mismatch += (in_d - out_d)
    
    # The number of paths needed is the maximum of start_mismatch and end_mismatch
    # Because each path can resolve one mismatch at the start and one at the end
    result = max(start_mismatch, end_mismatch)
    
    print(result)