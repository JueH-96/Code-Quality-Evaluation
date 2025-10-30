def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = list(map(int, data[1:N+1]))
    
    # The sum is minimized when the middle two elements are as close as possible to the average of the first and last
    # So, the sum is X[0] + X[-1] + 2 * (X[0] + X[-1]) / 2 * (N-2) / 2
    # Wait, no. Let's think differently.
    
    # The minimal sum is achieved when the middle N-2 elements are arranged such that they are symmetric around the midpoint of the first and last elements.
    # So, the sum is X[0] + X[-1] + (N-2) * (X[0] + X[-1]) / 2
    
    # Because the sum of the middle N-2 elements is (N-2) * (X[0] + X[-1]) / 2
    
    total = X[0] + X[-1] + (N-2) * (X[0] + X[-1]) / 2
    print(int(total))

if __name__ == "__main__":
    main()