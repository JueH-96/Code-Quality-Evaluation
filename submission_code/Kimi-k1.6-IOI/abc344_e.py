def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    A = list(map(int, data[idx:idx+N]))
    idx +=N
    Q = int(data[idx])
    idx +=1
    
    prev = {}
    next_ = {}
    for i in range(N):
        if i >0:
            prev[A[i]] = A[i-1]
        if i < N-1:
            next_[A[i]] = A[i+1]
    head = A[0]
    tail = A[-1]
    
    for _ in range(Q):
        query = data[idx]
        idx +=1
        if query == '1':
            x = int(data[idx])
            y = int(data[idx+1])
            idx +=2
            current_next = next_.get(x, None)
            next_[x] = y
            prev[y] = x
            next_[y] = current_next
            if current_next is not None:
                prev[current_next] = y
            else:
                tail = y
        else:
            x = int(data[idx])
            idx +=1
            prev_x = prev.get(x, None)
            next_x = next_.get(x, None)
            if prev_x is not None:
                next_[prev_x] = next_x
            else:
                head = next_x
            if next_x is not None:
                prev[next_x] = prev_x
            else:
                tail = prev_x
    
    result = []
    current = head
    while current is not None:
        result.append(str(current))
        current = next_.get(current, None)
    print(' '.join(result))
    
if __name__ == "__main__":
    main()