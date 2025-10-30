import sys
input = sys.stdin.read

def process_operations():
    data = input().split()
    N = int(data[0])
    S = list(data[1])
    Q = int(data[2])
    operations = data[3:]
    
    for i in range(Q):
        t = int(operations[i * 3])
        if t == 1:
            x = int(operations[i * 3 + 1]) - 1
            c = operations[i * 3 + 2]
            S[x] = c
        elif t == 2:
            S = [char.lower() for char in S]
        elif t == 3:
            S = [char.upper() for char in S]
    
    print("".join(S))

process_operations()