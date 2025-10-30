# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = []
    index = 1
    for i in range(1, N+1):
        row = list(map(int, data[index:index+i]))
        A.append(row)
        index += i
    
    current = 1
    for j in range(1, N+1):
        if j <= current:
            current = A[current-1][j-1]
        else:
            current = A[j-1][current-1]
    
    print(current)

if __name__ == "__main__":
    main()