# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    index = 1
    
    P = []
    for i in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        P.append((A, B, i + 1))
        index += 2
    
    Q = []
    for i in range(N):
        C = int(data[index])
        D = int(data[index + 1])
        Q.append((C, D, i + 1))
        index += 2
    
    # Sort P and Q by their x-coordinates
    P.sort()
    Q.sort()
    
    # Create the permutation R based on the sorted order of Q
    R = [q[2] for q in Q]
    
    # Output the permutation R
    print(" ".join(map(str, R)))

if __name__ == "__main__":
    main()