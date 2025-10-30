# YOUR CODE HERE
import numpy as np, sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx +=1
    A = np.array(list(map(int, data[idx:idx+N])), dtype=np.int64); idx +=N
    M = int(data[idx]); idx +=1
    B = np.array(list(map(int, data[idx:idx+M])), dtype=np.int64); idx +=M
    L = int(data[idx]); idx +=1
    C = np.array(list(map(int, data[idx:idx+L])), dtype=np.int64); idx +=L
    Q = int(data[idx]); idx +=1
    X = np.array(list(map(int, data[idx:idx+Q])), dtype=np.int64); idx +=Q
    
    # Compute all possible a + b + c
    # To handle large computations efficiently, break it down
    # First compute a + b
    AB = A[:, None] + B[None, :]  # Shape N x M
    # Then add c
    # Total possible sums: N * M * L <= 1e6
    # Flatten all sums
    S = AB[:, :, None] + C[None, None, :]  # Shape N x M x L
    S = S.ravel()
    # Sort the sums
    S_sorted = np.sort(S)
    # For each query X, use searchsorted
    indices = np.searchsorted(S_sorted, X)
    # Check if X is present
    found = (indices < len(S_sorted)) & (S_sorted[indices] == X)
    # Prepare answers
    answers = np.where(found, 'Yes', 'No')
    # Print the answers
    print('
'.join(answers))

if __name__ == "__main__":
    main()