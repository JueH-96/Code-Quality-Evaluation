import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(Q):
        if input[ptr] == '1':
            x = int(input[ptr+1])
            y = int(input[ptr+2])
            queries.append((1, x, y))
            ptr += 3
        else:
            x = int(input[ptr+1])
            queries.append((2, x))
            ptr += 2
    
    nodes = {}
    prev = None
    head = A[0]
    for i in range(len(A)):
        current = A[i]
        next_node = A[i+1] if i < len(A)-1 else None
        nodes[current] = (prev, next_node)
        prev = current
    
    for q in queries:
        if q[0] == 1:
            x, y = q[1], q[2]
            prev_x, next_x = nodes[x]
            nodes[x] = (prev_x, y)
            nodes[y] = (x, next_x)
            if next_x is not None:
                nodes[next_x] = (y, nodes[next_x][1])
        else:
            x = q[1]
            prev_x, next_x = nodes[x]
            if prev_x is not None:
                nodes[prev_x] = (nodes[prev_x][0], next_x)
            else:
                head = next_x
            if next_x is not None:
                nodes[next_x] = (prev_x, nodes[next_x][1])
            del nodes[x]
    
    result = []
    current = head
    while current is not None:
        result.append(str(current))
        current = nodes[current][1]
    print(' '.join(result))

if __name__ == '__main__':
    main()