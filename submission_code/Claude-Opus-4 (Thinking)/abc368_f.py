def calculate_grundy(max_val):
    grundy = [0] * (max_val + 1)
    
    for i in range(2, max_val + 1):
        reachable = set()
        
        # Find all divisors
        for j in range(1, int(i**0.5) + 1):
            if i % j == 0:
                if j < i:  # j is a proper divisor
                    reachable.add(grundy[j])
                k = i // j
                if k < i:  # k is a proper divisor
                    reachable.add(grundy[k])
        
        # Calculate mex
        mex = 0
        while mex in reachable:
            mex += 1
        
        grundy[i] = mex
    
    return grundy

# Read input
n = int(input())
a = list(map(int, input().split()))

# Calculate Grundy numbers
max_val = max(a)
grundy = calculate_grundy(max_val)

# Calculate XOR of Grundy numbers
xor_sum = 0
for num in a:
    xor_sum ^= grundy[num]

# Determine winner
if xor_sum != 0:
    print("Anna")
else:
    print("Bruno")