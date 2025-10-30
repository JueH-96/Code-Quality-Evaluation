# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    N, M = map(int, data.split())
    
    # Calculate the sum X = sum(N^i for i in range(M+1))
    X = 0
    current_power = 1  # This is N^0 initially which is 1
    
    for i in range(M + 1):
        X += current_power
        if X > 10**9:
            print("inf")
            return
        current_power *= N
    
    print(X)

if __name__ == "__main__":
    main()