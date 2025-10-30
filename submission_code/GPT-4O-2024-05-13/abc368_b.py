# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    operations = 0
    
    while sum(a > 0 for a in A) > 1:
        A.sort(reverse=True)
        if A[0] > 0 and A[1] > 0:
            A[0] -= 1
            A[1] -= 1
            operations += 1
    
    print(operations)

if __name__ == "__main__":
    main()