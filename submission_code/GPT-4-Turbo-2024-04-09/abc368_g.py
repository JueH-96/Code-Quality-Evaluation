import sys
input = sys.stdin.read

def main():
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    idx += 1
    
    A = list(map(int, data[idx:idx+N]))
    idx += N
    
    B = list(map(int, data[idx:idx+N]))
    idx += N
    
    Q = int(data[idx])
    idx += 1
    
    queries = data[idx:]
    
    results = []
    query_idx = 0
    
    while query_idx < len(queries):
        query_type = int(queries[query_idx])
        if query_type == 1:
            i = int(queries[query_idx + 1]) - 1
            x = int(queries[query_idx + 2])
            A[i] = x
            query_idx += 3
        elif query_type == 2:
            i = int(queries[query_idx + 1]) - 1
            x = int(queries[query_idx + 2])
            B[i] = x
            query_idx += 3
        elif query_type == 3:
            l = int(queries[query_idx + 1]) - 1
            r = int(queries[query_idx + 2]) - 1
            v = 0
            for i in range(l, r + 1):
                add_v = v + A[i]
                mul_v = v * B[i]
                v = max(add_v, mul_v)
            results.append(v)
            query_idx += 3
    
    for result in results:
        print(result)

main()