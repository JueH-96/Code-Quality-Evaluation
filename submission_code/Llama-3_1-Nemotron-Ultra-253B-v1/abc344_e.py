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
            x = int(data[ptr + 1])
            y = int(data[ptr + 2])
            queries.append((1, x, y))
            ptr += 3
        else:
            x = int(data[ptr + 1])
            queries.append((2, x))
            ptr += 2

    prev = {}
    next_ = {}
    current_elements = set(A)
    for i in range(len(A)):
        elem = A[i]
        if i > 0:
            prev[elem] = A[i - 1]
        else:
            prev[elem] = None
        if i < len(A) - 1:
            next_[elem] = A[i + 1]
        else:
            next_[elem] = None

    for q in queries:
        if q[0] == 1:
            x, y = q[1], q[2]
            old_next = next_.get(x, None)
            prev[y] = x
            next_[y] = old_next
            next_[x] = y
            if old_next is not None:
                prev[old_next] = y
            current_elements.add(y)
        else:
            x = q[1]
            p = prev.get(x, None)
            n = next_.get(x, None)
            if p is not None:
                next_[p] = n
            if n is not None:
                prev[n] = p
            current_elements.remove(x)

    head = None
    for elem in current_elements:
        if prev.get(elem, None) is None:
            head = elem
            break

    result = []
    current = head
    while current is not None:
        result.append(str(current))
        current = next_.get(current, None)
    print(' '.join(result))

if __name__ == '__main__':
    main()