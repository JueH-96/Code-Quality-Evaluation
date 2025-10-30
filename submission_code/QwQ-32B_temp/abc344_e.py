import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    prev = {}
    next_ = {}
    if N == 0:
        pass  # According to constraints, N >=1
    head = A[0]
    tail = A[-1]
    for i in range(N):
        current = A[i]
        if i == 0:
            prev[current] = None
        else:
            prev[current] = A[i-1]
        if i == N-1:
            next_[current] = None
        else:
            next_[current] = A[i+1]
    
    Q = int(sys.stdin.readline())
    for _ in range(Q):
        parts = sys.stdin.readline().split()
        if parts[0] == '1':
            x = int(parts[1])
            y = int(parts[2])
            current_next = next_[x]
            prev[y] = x
            next_[y] = current_next
            next_[x] = y
            if current_next is not None:
                prev[current_next] = y
            if x == tail:
                tail = y
        else:
            x = int(parts[1])
            prev_x = prev[x]
            next_x = next_[x]
            if prev_x is not None:
                next_[prev_x] = next_x
            else:
                head = next_x
            if next_x is not None:
                prev[next_x] = prev_x
            else:
                tail = prev_x
            del prev[x]
            del next_[x]
    
    result = []
    current = head
    while current is not None:
        result.append(str(current))
        current = next_[current]
    print(' '.join(result))

if __name__ == "__main__":
    main()