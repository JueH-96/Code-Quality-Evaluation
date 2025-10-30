def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    x_list = list(map(int, input[ptr:ptr+Q]))
    ptr += Q
    
    # Initialize toggles for each element
    toggles = [[] for _ in range(N + 1)]  # 1-based indexing
    for idx in range(Q):
        x = x_list[idx]
        toggles[x].append(idx + 1)  # 1-based query index
    
    # Compute s array and prefix sums
    s = [0] * (Q + 1)  # s[0] is dummy, s[1] to s[Q]
    current_size = 0
    present = [False] * (N + 1)  # 1-based elements
    
    for i in range(1, Q + 1):
        x = x_list[i - 1]
        if present[x]:
            current_size -= 1
            present[x] = False
        else:
            current_size += 1
            present[x] = True
        s[i] = current_size
    
    # Compute prefix sums
    pre = [0] * (Q + 1)
    for i in range(1, Q + 1):
        pre[i] = pre[i - 1] + s[i]
    
    # Calculate the result for each element
    result = [0] * (N + 1)  # result[0] unused
    for j in range(1, N + 1):
        ts = toggles[j]
        total = 0
        for m in range(0, len(ts), 2):
            start = ts[m]
            if m + 1 < len(ts):
                end = ts[m + 1] - 1
            else:
                end = Q
            if start > end:
                continue
            total += pre[end] - pre[start - 1]
        result[j] = total
    
    print(' '.join(map(str, result[1:N+1])))

if __name__ == "__main__":
    main()