import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:2+N-1]))
    
    A_sorted = sorted(A)
    sum_A = sum(A_sorted)
    s1 = A_sorted[0]
    sN_1 = A_sorted[-1]
    
    min_AN = -1
    
    for AN in range(0, 101):
        if AN <= s1:
            S_final = sum_A - sN_1
        elif AN >= sN_1:
            S_final = sum_A - s1
        else:
            S_final = sum_A - s1 - sN_1 + AN
        if S_final >= X:
            if min_AN == -1 or AN < min_AN:
                min_AN = AN
    print(min_AN)

if __name__ == '__main__':
    main()