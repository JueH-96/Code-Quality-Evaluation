def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = list(map(int, input[1:N+1]))
    
    initial_sum = sum(X)
    gains = 0
    
    for i in range(N - 3):
        sum_ab = X[i+1] + X[i+2]
        outer_sum = X[i] + X[i+3]
        if outer_sum < sum_ab:
            gains += 2 * (sum_ab - outer_sum)
    
    print(initial_sum - gains)

if __name__ == "__main__":
    main()