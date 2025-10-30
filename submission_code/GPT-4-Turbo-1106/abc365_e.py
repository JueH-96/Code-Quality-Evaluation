def calculate_xor_sum(N, A):
    # Initialize the result
    result = 0
    
    # Calculate the number of times each bit appears in the XOR sum
    for i in range(30):  # Since 1 <= A_i <= 10^8, we only need to consider 30 bits
        bit_count = 0
        total = 0
        for j in range(N):
            if A[j] & (1 << i):
                bit_count += 1
            else:
                total += bit_count * (bit_count + 1) // 2
                bit_count = 0
        total += bit_count * (bit_count + 1) // 2
        result += total * (1 << i)
    
    return result

# Read input
N = int(input().strip())
A = list(map(int, input().strip().split()))

# Calculate and print the result
print(calculate_xor_sum(N, A))