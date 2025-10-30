import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    # First line values
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    
    # Sequence A starts at index 3
    A = list(map(int, data[3:3 + N]))
    
    # Insert X after the K-th element (1-indexed)
    B = A[:K] + [X] + A[K:]
    
    print(' '.join(map(str, B)))

if __name__ == "__main__":
    main()