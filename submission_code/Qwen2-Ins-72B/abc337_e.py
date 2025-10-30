import math

def find_spoiled_juice():
    N = int(input())
    M = math.ceil(math.log2(N))
    
    print(M)
    print flush()
    
    binary = ['0'] * M
    for i in range(M):
        for j in range(2**i, N, 2**(i+1)):
            print(1, j+1)
            print flush()
        response = raw_input()
        binary[i] = response[i]
    
    X = int(''.join(binary), 2)
    print(X+1)
    print flush()

find_spoiled_juice()