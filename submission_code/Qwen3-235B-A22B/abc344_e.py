import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    prev = dict()
    next = dict()
    for i in range(N):
        current = A[i]
        if i == 0:
            prev[current] = None
        else:
            prev[current] = A[i-1]
        if i == N-1:
            next[current] = None
        else:
            next[current] = A[i+1]
    head = A[0]
    tail = A[-1]

    Q = int(input[ptr])
    ptr += 1

    for _ in range(Q):
        t = int(input[ptr])
        if t == 1:
            x = int(input[ptr+1])
            y = int(input[ptr+2])
            ptr +=3
            z = next[x]
            prev[y] = x
            next[y] = z
            next[x] = y
            if z is not None:
                prev[z] = y
            else:
                tail = y
        else:
            x = int(input[ptr+1])
            ptr +=2
            prev_x = prev[x]
            next_x = next[x]
            if prev_x is not None:
                next[prev_x] = next_x
            else:
                head = next_x
            if next_x is not None:
                prev[next_x] = prev_x
            else:
                tail = prev_x
            del prev[x]
            del next[x]
    
    res = []
    current = head
    while current is not None:
        res.append(str(current))
        current = next.get(current)
    print(' '.join(res))

if __name__ == "__main__":
    main()