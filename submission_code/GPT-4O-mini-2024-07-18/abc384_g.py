def calculate_sum_of_absolute_differences(N, A, B, K, queries):
    results = []
    
    # Precompute the absolute differences
    abs_diff = [[abs(A[i] - B[j]) for j in range(N)] for i in range(N)]
    
    for x, y in queries:
        total_sum = sum(abs_diff[i][j] for i in range(x) for j in range(y))
        results.append(total_sum)
    
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    A = list(map(int, data[1].split()))
    B = list(map(int, data[2].split()))
    K = int(data[3])
    
    queries = [tuple(map(int, line.split())) for line in data[4:4 + K]]
    
    results = calculate_sum_of_absolute_differences(N, A, B, K, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()