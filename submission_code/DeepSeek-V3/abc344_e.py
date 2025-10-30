# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N
    Q = int(data[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        if data[idx] == '1':
            x = int(data[idx+1])
            y = int(data[idx+2])
            queries.append(('1', x, y))
            idx += 3
        else:
            x = int(data[idx+1])
            queries.append(('2', x))
            idx += 2
    
    # Initialize the linked list
    prev = {}
    next_ = {}
    for i in range(N):
        if i == 0:
            prev[A[i]] = None
        else:
            prev[A[i]] = A[i-1]
        if i == N-1:
            next_[A[i]] = None
        else:
            next_[A[i]] = A[i+1]
    
    for query in queries:
        if query[0] == '1':
            x, y = query[1], query[2]
            # Insert y after x
            next_y = next_[x]
            next_[x] = y
            prev[y] = x
            next_[y] = next_y
            if next_y is not None:
                prev[next_y] = y
        else:
            x = query[1]
            # Remove x
            p = prev[x]
            n = next_[x]
            if p is not None:
                next_[p] = n
            if n is not None:
                prev[n] = p
            del prev[x]
            del next_[x]
    
    # Reconstruct the sequence
    current = None
    for key in prev:
        if prev[key] is None:
            current = key
            break
    result = []
    while current is not None:
        result.append(current)
        current = next_[current]
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()