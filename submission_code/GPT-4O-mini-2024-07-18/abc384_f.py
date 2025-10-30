def f(x):
    while x % 2 == 0:
        x //= 2
    return x

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    total_sum = 0
    
    # Calculate the sum of f(A[i] + A[j]) for all i <= j
    for i in range(N):
        for j in range(i, N):
            total_sum += f(A[i] + A[j])
    
    print(total_sum)

if __name__ == "__main__":
    main()