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
    
    K = int(data[idx])
    idx += 1
    queries = []
    for _ in range(K):
        X_k = int(data[idx])
        Y_k = int(data[idx+1])
        queries.append((X_k, Y_k))
        idx += 2
    
    results = []
    for X_k, Y_k in queries:
        sum_value = 0
        for i in range(X_k):
            for j in range(Y_k):
                sum_value += abs(A[i] - B[j])
        results.append(sum_value)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()