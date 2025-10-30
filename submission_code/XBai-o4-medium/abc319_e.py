import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    Y = int(input[ptr])
    ptr += 1
    
    buses = []
    for _ in range(N-1):
        p = int(input[ptr])
        t = int(input[ptr+1])
        buses.append((p, t))
        ptr += 2
    
    mod = 840
    base_time = [i for i in range(mod)]
    
    for p, t in buses:
        base_time = [ ((x + p - 1) // p) * p + t for x in base_time ]
    
    Q = int(input[ptr])
    ptr += 1
    output = []
    for _ in range(Q):
        q_i = int(input[ptr])
        ptr += 1
        arrival_time_1 = q_i + X
        r = arrival_time_1 % mod
        arrival_time_N = base_time[r] - r + arrival_time_1
        total = arrival_time_N + Y
        output.append(str(total))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()