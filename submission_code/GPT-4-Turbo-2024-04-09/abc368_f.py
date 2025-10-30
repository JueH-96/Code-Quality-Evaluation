def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Since each A_i >= 2, we need to determine the Grundy number (nimber) for each A_i
    # The Grundy number for a number n is 0 if n is a terminal position (no moves possible)
    # which is not the case for any n >= 2 in this problem since they all have divisors < themselves.
    
    # We need to calculate the Grundy numbers for all numbers up to 100000
    # Let's use a sieve-like approach to calculate the minimum excluded value (mex) for each number.
    
    MAX_A = 100000
    grundy = [0] * (MAX_A + 1)
    
    # Calculate Grundy numbers using the definition:
    # Grundy(n) = mex({Grundy(d) | d is a proper divisor of n})
    # We start from 2 up to MAX_A
    for i in range(2, MAX_A + 1):
        divisors = set()
        # Collect grundy numbers of all proper divisors
        for j in range(1, int(i**0.5) + 1):
            if i % j == 0:
                if j < i:
                    divisors.add(grundy[j])
                if i // j < i and i // j != j:
                    divisors.add(grundy[i // j])
        
        # Calculate mex (minimum excluded value)
        mex_value = 0
        while mex_value in divisors:
            mex_value += 1
        
        grundy[i] = mex_value
    
    # Now calculate the nim-sum (xor sum) of the Grundy numbers of the given sequence
    nim_sum = 0
    for a in A:
        nim_sum ^= grundy[a]
    
    # If the nim-sum is 0, the position is losing for the first player (Anna)
    # Otherwise, it's winning for the first player (Anna)
    if nim_sum == 0:
        print("Bruno")
    else:
        print("Anna")

if __name__ == "__main__":
    main()