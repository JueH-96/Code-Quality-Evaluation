# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Precompute f(x) for all x from 1 to 2 * 10^7
    max_possible_sum = 2 * 10**7
    f = [0] * (max_possible_sum + 1)
    
    for x in range(1, max_possible_sum + 1):
        if x % 2 == 0:
            f[x] = f[x // 2]
        else:
            f[x] = x
    
    # Calculate the sum of f(A_i + A_j) for all i <= j
    total_sum = 0
    
    for i in range(N):
        for j in range(i, N):
            total_sum += f[A[i] + A[j]]
    
    print(total_sum)

main()