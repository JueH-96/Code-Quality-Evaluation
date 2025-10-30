# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:]))
    
    A = []
    
    for i in range(1, N + 1):
        position = P[i - 1] - 1
        A.insert(position, i)
    
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()