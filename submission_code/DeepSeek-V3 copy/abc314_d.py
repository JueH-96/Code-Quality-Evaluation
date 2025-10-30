def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = list(data[1])
    Q = int(data[2])
    
    operations = []
    index = 3
    for _ in range(Q):
        t = int(data[index])
        x = int(data[index+1])
        c = data[index+2]
        operations.append((t, x, c))
        index += 3
    
    last_case_operation = 0
    for op in operations:
        if op[0] == 2 or op[0] == 3:
            last_case_operation = op[0]
    
    for op in operations:
        t, x, c = op
        if t == 1:
            S[x-1] = c
        elif t == 2:
            if last_case_operation == 2:
                continue
            for i in range(N):
                if S[i].isupper():
                    S[i] = S[i].lower()
            last_case_operation = 2
        elif t == 3:
            if last_case_operation == 3:
                continue
            for i in range(N):
                if S[i].islower():
                    S[i] = S[i].upper()
            last_case_operation = 3
    
    print(''.join(S))

if __name__ == "__main__":
    main()