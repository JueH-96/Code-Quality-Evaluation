import math

def find_spoiled_juice(N):
    # Calculate the minimum number of friends needed
    M = math.ceil(math.log2(N))
    print(M)
    
    # Distribute the juice bottles to friends
    for i in range(M):
        print_bin = []
        for j in range(1, N + 1):
            if (j >> i) & 1:
                print_bin.append(j)
        print(len(print_bin), *print_bin)
    
    # Receive the result from the judge
    result = input().strip()
    
    # Calculate the spoiled juice bottle number
    spoiled_juice = 0
    for i in range(M):
        spoiled_juice = (spoiled_juice << 1) | int(result[i])
    
    print(spoiled_juice)

N = int(input())
find_spoiled_juice(N)