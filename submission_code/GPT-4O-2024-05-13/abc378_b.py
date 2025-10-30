# YOUR CODE HERE
import sys
input = sys.stdin.read

def next_collection_day(N, garbage_info, Q, queries):
    results = []
    for t, d in queries:
        q, r = garbage_info[t-1]
        if d % q <= r:
            next_day = d - (d % q) + r
        else:
            next_day = d - (d % q) + q + r
        results.append(next_day)
    return results

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    
    garbage_info = []
    for _ in range(N):
        q = int(data[index])
        r = int(data[index + 1])
        garbage_info.append((q, r))
        index += 2
    
    Q = int(data[index])
    index += 1
    
    queries = []
    for _ in range(Q):
        t = int(data[index])
        d = int(data[index + 1])
        queries.append((t, d))
        index += 2
    
    results = next_collection_day(N, garbage_info, Q, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()