def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    p = int(data[1])
    A = []
    index = 2
    for i in range(N):
        A.append([int(data[index + j]) for j in range(N)])
        index += N
    
    # Fermat's Little Theorem: a^p ≡ a (mod p) for any integer a when p is prime
    # Therefore, B^p ≡ B (mod p) for any matrix B when p is prime
    # We need to sum all possible B matrices and then take each element mod p
    
    # Count the number of zeros in each position
    zero_count = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                zero_count[i][j] = 1
    
    # Calculate (p-1)^K where K is the number of zeros in the matrix
    from functools import reduce
    total_zeros = sum(sum(row) for row in zero_count)
    if total_zeros == 0:
        # No zero in the matrix, just output the matrix itself mod p
        for row in A:
            print(" ".join(str(x % p) for x in row))
        return
    
    # Calculate (p-1)^total_zeros
    mod_powers = (p - 1) ** total_zeros
    
    # Sum all possible B matrices
    result = [[0] * N for _ in range(N)]
    
    # We iterate through each cell
    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                # This cell was zero, it contributes (1 + 2 + ... + (p-1)) * (p-1)^(total_zeros-1) to each cell
                # Sum of 1 to p-1 is p*(p-1)/2
                sum_contribution = (p * (p - 1) // 2) * mod_powers // (p - 1)
            else:
                # This cell was non-zero, it contributes A[i][j] * (p-1)^total_zeros to each cell
                sum_contribution = A[i][j] * mod_powers
            
            result[i][j] = sum_contribution % p
    
    # Print the result matrix
    for row in result:
        print(" ".join(str(x) for x in row))

if __name__ == "__main__":
    main()