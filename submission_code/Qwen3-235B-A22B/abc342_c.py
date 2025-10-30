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
    
    mapping = list('abcdefghijklmnopqrstuvwxyz')
    
    for _ in range(Q):
        c = data[idx]
        d = data[idx+1]
        idx += 2
        for i in range(26):
            if mapping[i] == c:
                mapping[i] = d
    
    res = []
    for ch in S:
        res.append(mapping[ord(ch) - ord('a')])
    print(''.join(res))

if __name__ == '__main__':
    main()