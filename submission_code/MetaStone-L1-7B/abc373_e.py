def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    K = int(data[idx+2])
    idx +=3
    A = list(map(int, data[idx:idx+N]))
    idx +=N
    R = list(map(int, data[idx:idx+N]))
    idx +=N

    s = [A[i] + R[i] for i in range(N)]
    global_max = max(s)
    count_global_max = s.count(global_max)

    if count_global_max > 1:
        second_max = global_max
    else:
        second_max = max(x for x in s if x != global_max)

    result = []
    for i in range(N):
        if s[i] == global_max:
            if count_global_max > 1:
                max_val = global_max
            else:
                max_val = second_max
        else:
            max_val = global_max
        X_i = max(0, max_val - A[i] + 1)
        result.append(str(X_i))
    
    print(' '.join(result))

if __name__ == '__main__':
    main()