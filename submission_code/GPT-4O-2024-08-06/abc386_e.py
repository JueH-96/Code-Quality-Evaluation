def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # Step 1: Build the basis
    basis = []
    
    for num in A:
        # Try to reduce num using the current basis
        for b in basis:
            num = min(num, num ^ b)
        
        # If num is not zero, it is a new basis element
        if num != 0:
            basis.append(num)
            # Keep the basis sorted by the highest bit position
            basis.sort(reverse=True)
    
    # Step 2: Calculate the maximum XOR using the basis
    max_xor = 0
    for b in basis:
        max_xor = max(max_xor, max_xor ^ b)
    
    print(max_xor)