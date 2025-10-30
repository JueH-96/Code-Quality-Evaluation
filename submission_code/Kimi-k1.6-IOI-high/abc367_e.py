import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    X = list(map(lambda x: int(x)-1, input[ptr:ptr+N]))
    ptr += N
    A = list(map(int, input[ptr:ptr+N]))
    
    if K == 0:
        print(' '.join(map(str, A)))
        return
    
    max_level = 60
    # Initialize the lift table
    lift = [X]
    for _ in range(max_level):
        prev = lift[-1]
        next_level = [prev[prev[i]] for i in range(N)]
        lift.append(next_level)
    
    result = []
    for i in range(N):
        pos = i
        remaining = K
        for j in range(max_level, -1, -1):
            if remaining >= (1 << j):
                pos = lift[j][pos]
                remaining -= (1 << j)
        result.append(A[pos])
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()