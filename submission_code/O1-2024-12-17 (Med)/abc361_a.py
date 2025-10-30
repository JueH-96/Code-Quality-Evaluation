def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    N, K, X = map(int, data[:3])         # Read N, K, X
    A = list(map(int, data[3:3+N]))      # Read the sequence A
    
    B = A[:K] + [X] + A[K:]             # Insert X after the K-th element
    print(*B)                           # Print the resulting sequence

# Do not forget to call main()
main()