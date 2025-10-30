import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    last_position = {}
    B = []
    
    for i in range(N):
        a_i = A[i]
        if a_i in last_position:
            B.append(last_position[a_i])
        else:
            B.append(-1)
        last_position[a_i] = i + 1
    
    print(" ".join(map(str, B)))