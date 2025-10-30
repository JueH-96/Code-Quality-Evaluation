import sys

def move(X, Y, F, N):
    if X == Y:
        return 0
    if X < Y:
        d_clock = Y - X
        if X < F < Y:
            return N - d_clock
        else:
            return d_clock
    else:
        d_clock = Y - X + N
        if F > X or F < Y:
            return N - d_clock
        else:
            return d_clock

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
    
    first_line = data[0].split()
    N = int(first_line[0])
    Q = int(first_line[1])
    
    instructions = []
    for i in range(1, Q + 1):
        parts = data[i].split()
        H = parts[0]
        T = int(parts[1])
        instructions.append((H, T))
        
    l = 1
    r = 2
    total_ops = 0
    
    for inst in instructions:
        H, T = inst
        if H == 'L':
            cost = move(l, T, r, N)
            total_ops += cost
            l = T
        else:
            cost = move(r, T, l, N)
            total_ops += cost
            r = T
            
    print(total_ops)

if __name__ == "__main__":
    main()