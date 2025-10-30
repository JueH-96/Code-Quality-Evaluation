import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    
    queries = []
    index = 3
    for _ in range(Q):
        l = int(data[index]) - 1  # Convert to 0-based index
        r = int(data[index + 1]) - 1  # Convert to 0-based index
        queries.append((l, r))
        index += 2
    
    # Precompute the number of consecutive equal characters up to each position
    consecutive_counts = [0] * N
    for i in range(1, N):
        consecutive_counts[i] = consecutive_counts[i - 1]
        if S[i] == S[i - 1]:
            consecutive_counts[i] += 1
    
    # Answer each query using the precomputed results
    results = []
    for l, r in queries:
        if l == 0:
            results.append(consecutive_counts[r])
        else:
            results.append(consecutive_counts[r] - consecutive_counts[l])
    
    # Print all results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()