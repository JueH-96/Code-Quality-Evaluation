def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr +=1
    S = list(data[ptr])
    ptr +=1
    Q = int(data[ptr])
    ptr +=1
    
    last_op_num = [0] * N
    last_type2_or_3_op = 0
    current_mode = None
    
    for i in range(1, Q+1):
        t = int(data[ptr])
        ptr +=1
        x = int(data[ptr])
        ptr +=1
        c = data[ptr]
        ptr +=1
        
        if t == 1:
            pos = x-1
            S[pos] = c
            last_op_num[pos] = i
        elif t == 2:
            last_type2_or_3_op = i
            current_mode = 'lower'
        elif t ==3:
            last_type2_or_3_op = i
            current_mode = 'upper'
    
    result = []
    for idx in range(N):
        if last_op_num[idx] > last_type2_or_3_op:
            result.append(S[idx])
        else:
            if current_mode == 'lower':
                result.append(S[idx].lower())
            elif current_mode == 'upper':
                result.append(S[idx].upper())
            else:
                result.append(S[idx])
    print(''.join(result))

if __name__ == "__main__":
    main()