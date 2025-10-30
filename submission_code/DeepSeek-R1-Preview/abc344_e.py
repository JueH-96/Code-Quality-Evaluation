import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr + N]))
    ptr += N
    Q = int(data[ptr])
    ptr += 1
    queries = []
    for _ in range(Q):
        if data[ptr] == '1':
            x = int(data[ptr+1])
            y = int(data[ptr+2])
            queries.append((1, x, y))
            ptr += 3
        else:
            x = int(data[ptr+1])
            queries.append((2, x))
            ptr += 2

    next_node = {}
    prev_node = {}
    head = A[0]
    tail = A[-1]
    for i in range(len(A)):
        current = A[i]
        if i == 0:
            prev_node[current] = None
        else:
            prev_node[current] = A[i-1]
        if i == len(A) - 1:
            next_node[current] = None
        else:
            next_node[current] = A[i+1]

    for q in queries:
        if q[0] == 1:
            x, y = q[1], q[2]
            z = next_node.get(x, None)
            next_node[x] = y
            prev_node[y] = x
            next_node[y] = z
            if z is not None:
                prev_node[z] = y
            else:
                tail = y
        else:
            x = q[1]
            p = prev_node.get(x, None)
            n = next_node.get(x, None)
            if p is not None:
                next_node[p] = n
            else:
                head = n
            if n is not None:
                prev_node[n] = p
            else:
                tail = p

    result = []
    current = head
    while current is not None:
        result.append(str(current))
        current = next_node.get(current, None)
    print(' '.join(result))

if __name__ == '__main__':
    main()