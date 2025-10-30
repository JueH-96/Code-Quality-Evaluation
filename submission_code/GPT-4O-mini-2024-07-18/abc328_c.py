def count_repeated_pairs(N, Q, S, queries):
    # Precompute the count of adjacent equal characters
    count = [0] * (N + 1)
    
    for i in range(1, N):
        count[i] = count[i - 1] + (1 if S[i - 1] == S[i] else 0)
    
    results = []
    for l, r in queries:
        # Adjust indices for 0-based indexing
        results.append(count[r - 1] - count[l - 1])
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, Q = map(int, data[0].split())
    S = data[1]
    queries = [tuple(map(int, line.split())) for line in data[2:2 + Q]]
    
    results = count_repeated_pairs(N, Q, S, queries)
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()