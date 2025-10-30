# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    A = list(map(int, data[3:3+N]))
    
    result = []
    for i in range(N):
        A_i = A[i]
        if A_i < L:
            result.append(L)
        elif A_i > R:
            result.append(R)
        else:
            result.append(A_i)
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()