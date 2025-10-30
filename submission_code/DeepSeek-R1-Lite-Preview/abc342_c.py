def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    Q = int(data[2])
    
    operations = data[3: 3 + 2 * Q]
    
    mapping = {chr(i): chr(i) for i in range(ord('a'), ord('z') + 1)}
    
    for i in range(Q):
        c = operations[2 * i]
        d = operations[2 * i + 1]
        mapping[c] = d
    
    S_list = list(S)
    for i in range(N):
        S_list[i] = mapping[S_list[i]]
    
    print(''.join(S_list))

if __name__ == "__main__":
    main()