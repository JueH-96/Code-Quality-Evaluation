def main():
    import sys
    input = sys.stdin.readline

    # Read the input values
    N = int(input().strip())
    A = list(map(int, input().split()))
    
    # Check strictly increasing condition
    for i in range(N - 1):
        if A[i] >= A[i+1]:
            print("No")
            return
    print("Yes")
    
if __name__ == '__main__':
    main()