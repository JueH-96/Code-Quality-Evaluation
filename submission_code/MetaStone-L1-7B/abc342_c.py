def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    S = data[idx]
    idx += 1
    Q = int(data[idx])
    idx += 1
    operations = []
    for _ in range(Q):
        c_i = data[idx]
        d_i = data[idx + 1]
        operations.append((c_i, d_i))
        idx += 2

    # Process operations in reverse order to build the reverse mapping
    reverse_ops = reversed(operations)
    reverse_mapping = {}
    for op in reverse_ops:
        c_i, d_i = op
        reverse_mapping[c_i] = d_i

    result = []
    for c in S:
        current = c
        for op in reverse_ops:
            if current in reverse_mapping:
                current = reverse_mapping[current]
        result.append(current)
    
    print(''.join(result))

if __name__ == "__main__":
    main()