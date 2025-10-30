# YOUR CODE HERE
import sys

def process_queries(N, Q, queries):
    A = [0] * N
    S = set()
    
    for x in queries:
        x -= 1  # Adjust for 0-based indexing
        if x in S:
            S.remove(x)
        else:
            S.add(x)
        
        for j in S:
            A[j] += len(S)
    
    return A

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    queries = list(map(int, data[2:]))
    
    result = process_queries(N, Q, queries)
    print(" ".join(map(str, result)))