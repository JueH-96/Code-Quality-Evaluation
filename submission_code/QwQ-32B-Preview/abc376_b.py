def distance(A, B, C, N):
    if A == B:
        return 0
    clockwise = (B - A) % N
    if clockwise == 0:
        clockwise = N
    counterclockwise = N - clockwise
    if A < B:
        if A < C and C < B:
            return counterclockwise
        else:
            return clockwise
    else:
        if B < C and C < A:
            return counterclockwise
        else:
            return clockwise

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    instructions = data[2:]
    left = 1
    right = 2
    total_ops = 0
    for i in range(Q):
        H = instructions[2*i]
        T = int(instructions[2*i + 1])
        if H == 'L':
            ops = distance(left, T, right, N)
            total_ops += ops
            left = T
        else:
            ops = distance(right, T, left, N)
            total_ops += ops
            right = T
    print(total_ops)

if __name__ == '__main__':
    main()