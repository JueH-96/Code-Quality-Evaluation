def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    queries = []
    for i in range(Q):
        l = int(data[3 + 2*i])
        r = int(data[4 + 2*i])
        queries.append((l, r))
    
    # Build prefix array
    prefix = [0] * N
    for i in range(1, N):
        if S[i-1] == S[i]:
            prefix[i] = prefix[i-1] + 1
        else:
            prefix[i] = prefix[i-1]
    
    # Collect answers
    answers = []
    for l, r in queries:
        if r - l >= 1:
            ans = prefix[r-1] - prefix[l-1]
        else:
            ans = 0
        answers.append(str(ans))
    
    # Output all answers
    print('
'.join(answers))

if __name__ == '__main__':
    main()