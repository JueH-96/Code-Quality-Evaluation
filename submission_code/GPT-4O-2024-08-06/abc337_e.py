# YOUR CODE HERE
import sys
import math

def main():
    input = sys.stdin.read
    data = input().strip()
    N = int(data)
    
    # Calculate the number of friends needed
    M = math.ceil(math.log2(N))
    print(M)
    
    # Distribute bottles to friends based on binary representation
    for i in range(M):
        # Each friend i will taste bottles where the i-th bit is set
        bottles = []
        for j in range(1, N + 1):
            if (j >> i) & 1:
                bottles.append(j)
        
        # Output the bottles for the i-th friend
        print(len(bottles), ' '.join(map(str, bottles)))
    
    # Read the response from the judge
    S = input().strip()
    
    # Determine the spoiled bottle
    spoiled_bottle = 0
    for i in range(M):
        if S[i] == '1':
            spoiled_bottle |= (1 << i)
    
    # Output the spoiled bottle
    print(spoiled_bottle + 1)

main()