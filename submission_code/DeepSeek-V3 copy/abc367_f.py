import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+N]))
    idx += N
    
    # Precompute prefix sums for A and B
    # For each number from 1 to N, compute the cumulative count at each position
    prefix_A = [defaultdict(int) for _ in range(N+1)]
    for i in range(1, N+1):
        prefix_A[i] = prefix_A[i-1].copy()
        prefix_A[i][A[i-1]] += 1
    
    prefix_B = [defaultdict(int) for _ in range(N+1)]
    for i in range(1, N+1):
        prefix_B[i] = prefix_B[i-1].copy()
        prefix_B[i][B[i-1]] += 1
    
    for _ in range(Q):
        l = int(data[idx])
        r = int(data[idx+1])
        L = int(data[idx+2])
        R = int(data[idx+3])
        idx += 4
        
        # Get the counts for the subsequence in A
        count_A = defaultdict(int)
        for key in prefix_A[r]:
            count_A[key] = prefix_A[r][key] - prefix_A[l-1][key]
        
        # Get the counts for the subsequence in B
        count_B = defaultdict(int)
        for key in prefix_B[R]:
            count_B[key] = prefix_B[R][key] - prefix_B[L-1][key]
        
        # Check if the counts match
        if count_A == count_B:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()