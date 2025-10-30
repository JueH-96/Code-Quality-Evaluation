def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]

    stack = []
    delete_ranges = []
    for i in range(N):
        if S[i] == '(':
            stack.append(i)
        elif S[i] == ')':
            if stack and S[stack[-1]] == '(':
                start = stack.pop()
                delete_ranges.append((start, i))
    
    result = []
    current = 0
    range_idx = 0
    delete_len = len(delete_ranges)
    
    while current < N:
        if range_idx < delete_len and current == delete_ranges[range_idx][0]:
            end = delete_ranges[range_idx][1]
            current = end + 1
            range_idx += 1
        else:
            result.append(S[current])
            current += 1
    
    print(''.join(result))

if __name__ == "__main__":
    main()