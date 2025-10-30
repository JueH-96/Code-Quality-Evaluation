import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    
    points_P = []
    points_Q = []
    
    for i in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        points_P.append((A, B, i))
        index += 2
    
    for i in range(N):
        C = int(data[index])
        D = int(data[index + 1])
        points_Q.append((C, D, i))
        index += 2
    
    # Sort points by x-coordinate, and then by y-coordinate if x is the same
    points_P.sort()
    points_Q.sort()
    
    # We will map the sorted indices of P to the sorted indices of Q
    result = [0] * N
    for i in range(N):
        result[points_P[i][2]] = points_Q[i][2] + 1  # +1 because we need 1-based index
    
    print(" ".join(map(str, result)))