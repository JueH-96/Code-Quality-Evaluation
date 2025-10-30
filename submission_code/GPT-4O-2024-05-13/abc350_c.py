# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    operations = []
    
    for i in range(N):
        while A[i] != i + 1:
            j = A[i] - 1
            A[i], A[j] = A[j], A[i]
            operations.append((i + 1, j + 1))
    
    print(len(operations))
    for op in operations:
        print(op[0], op[1])

if __name__ == "__main__":
    main()