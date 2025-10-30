import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    A = list(map(int, data[3:]))
    
    # Insert X after the K-th element (1-based index)
    B = A[:K] + [X] + A[K:]
    
    # Print the result
    print(" ".join(map(str, B)))

if __name__ == "__main__":
    main()