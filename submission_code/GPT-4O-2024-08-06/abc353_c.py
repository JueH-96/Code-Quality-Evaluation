# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    MOD = 10**8
    total_sum = 0
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            total_sum += (A[i] + A[j]) % MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()