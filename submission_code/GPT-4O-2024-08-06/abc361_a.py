# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    A = list(map(int, data[3:]))
    
    # Create the sequence B by inserting X after the K-th element
    B = A[:K] + [X] + A[K:]
    
    # Print the sequence B
    print(' '.join(map(str, B)))

if __name__ == "__main__":
    main()