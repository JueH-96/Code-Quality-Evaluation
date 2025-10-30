def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1] if len(input) > 1 else ''
    
    stack = []
    pairs = []
    
    for i, c in enumerate(S):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    
    # Merge intervals
    merged = []
    for start, end in sorted(pairs):
        if not merged:
            merged.append([start, end])
        else:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])
    
    # Mark deleted characters
    deleted = [False] * N
    for s, e in merged:
        for i in range(s, e + 1):
            deleted[i] = True
    
    # Build result
    result = []
    for i in range(N):
        if not deleted[i]:
            result.append(S[i])
    
    print(''.join(result))

if __name__ == '__main__':
    main()