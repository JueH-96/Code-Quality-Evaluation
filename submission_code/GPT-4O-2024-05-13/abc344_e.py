# YOUR CODE HERE
import sys
input = sys.stdin.read

def process_queries(N, A, Q, queries):
    A = list(A)
    for query in queries:
        parts = query.split()
        if parts[0] == '1':
            x, y = int(parts[1]), int(parts[2])
            idx = A.index(x)
            A.insert(idx + 1, y)
        elif parts[0] == '2':
            x = int(parts[1])
            A.remove(x)
    return A

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    Q = int(data[N+1])
    queries = data[N+2:]
    
    result = process_queries(N, A, Q, queries)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()