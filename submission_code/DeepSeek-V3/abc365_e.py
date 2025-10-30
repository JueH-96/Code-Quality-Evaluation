# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute prefix XORs
    prefix_xor = [0] * (N + 1)
    for i in range(N):
        prefix_xor[i+1] = prefix_xor[i] ^ A[i]
    
    # Initialize the result
    result = 0
    
    # Iterate over each bit position
    for bit in range(60):  # Since A_i <= 10^8, 60 bits are sufficient
        mask = 1 << bit
        count = 0
        # Count the number of prefix_xor[j] with the bit set
        for j in range(N+1):
            if prefix_xor[j] & mask:
                count += 1
        # The number of subarrays with XOR having this bit set is count * (N+1 - count)
        result += mask * count * (N+1 - count)
    
    print(result)

if __name__ == "__main__":
    main()