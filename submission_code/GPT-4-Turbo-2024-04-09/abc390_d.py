def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Using Gaussian elimination in the vector space over GF(2) (binary field)
    # to find the basis of the space spanned by the given numbers.
    # This basis will allow us to determine the number of distinct XOR sums we can achieve.
    
    def gaussian_elimination_xor(basis):
        # Perform Gaussian elimination on the basis using XOR operations
        index = 0
        for i in range(60, -1, -1):  # Check each bit position from highest to lowest
            # Find a number in the basis with the i-th bit set
            for j in range(index, len(basis)):
                if (basis[j] >> i) & 1:
                    # Swap to make basis[index] have this bit set
                    basis[index], basis[j] = basis[j], basis[index]
                    break
            else:
                # No number with the i-th bit set
                continue
            
            # Now basis[index] has the i-th bit set, eliminate this bit from all other numbers
            for j in range(len(basis)):
                if j != index and ((basis[j] >> i) & 1):
                    basis[j] ^= basis[index]
            
            index += 1
            if index == len(basis):
                break
        
        # Filter out zero elements which might have been created during elimination
        basis = [x for x in basis if x != 0]
        return basis
    
    basis = gaussian_elimination_xor(A)
    
    # The number of distinct XOR sums we can achieve is 2^len(basis)
    # because each element in the basis can either be included in the XOR or not.
    print(2 ** len(basis))