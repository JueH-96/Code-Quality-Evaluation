# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    A = list(map(int, input().strip().split()))
    
    result = sum(A[i] * (2 ** i) for i in range(64))
    
    print(result)

if __name__ == "__main__":
    main()