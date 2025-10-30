import sys
input = sys.stdin.read

def solve():
    N = int(input().strip())
    
    # We need to determine the minimum number of friends (M) such that we can uniquely identify the spoiled bottle.
    # Using binary representation, we can uniquely identify each bottle using ceil(log2(N)) friends.
    # Each friend will test a subset of bottles corresponding to the set bits in the binary representation of the bottle numbers.
    
    # Calculate the number of friends needed (which is the number of bits needed to represent N bottles)
    M = N.bit_length()
    
    # Print the number of friends
    print(M)
    
    # Prepare the list of bottles each friend will test
    # Friend i (0-indexed) tests all bottles where the i-th bit in the binary representation of the bottle number is set
    for i in range(M):
        bottles = []
        for j in range(1, N + 1):
            if (j >> i) & 1:
                bottles.append(j)
        # Print the number of bottles and the list of bottles for this friend
        print(len(bottles), *bottles)
    
    # Flush to ensure the output is sent to the judge
    sys.stdout.flush()
    
    # Read the response from the judge, which is a string of length M
    response = input().strip()
    
    # Determine which bottle is spoiled based on the responses
    spoiled_bottle = 0
    for i in range(M):
        if response[i] == '1':
            spoiled_bottle |= (1 << i)
    
    # Print the spoiled bottle number
    print(spoiled_bottle)