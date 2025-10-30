# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Precompute powers of 10 modulo MOD
    powers_of_10 = [1] * 11
    for i in range(1, 11):
        powers_of_10[i] = (powers_of_10[i-1] * 10) % MOD
    
    result = 0
    
    # Iterate over all pairs (i, j) with i < j
    for i in range(N):
        for j in range(i + 1, N):
            x = A[i]
            y = A[j]
            
            # Calculate the number of digits in y
            len_y = len(str(y))
            
            # Calculate f(x, y) = x * 10^len_y + y
            f_xy = (x * powers_of_10[len_y] + y) % MOD
            
            # Add to result
            result = (result + f_xy) % MOD
    
    print(result)

main()