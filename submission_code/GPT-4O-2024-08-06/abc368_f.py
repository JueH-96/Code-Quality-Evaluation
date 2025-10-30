# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    MAX_A = 100000
    
    # Precompute Grundy numbers for all numbers from 2 to MAX_A
    grundy = [0] * (MAX_A + 1)
    
    for i in range(2, MAX_A + 1):
        divisors_grundy = set()
        for j in range(1, int(i**0.5) + 1):
            if i % j == 0:
                if j != i:
                    divisors_grundy.add(grundy[j])
                if i // j != i:
                    divisors_grundy.add(grundy[i // j])
        
        # Find the minimum excludant (mex)
        mex = 0
        while mex in divisors_grundy:
            mex += 1
        grundy[i] = mex
    
    # Calculate the XOR of all Grundy numbers of A
    xor_sum = 0
    for a in A:
        xor_sum ^= grundy[a]
    
    # Determine the winner
    if xor_sum == 0:
        print("Bruno")
    else:
        print("Anna")