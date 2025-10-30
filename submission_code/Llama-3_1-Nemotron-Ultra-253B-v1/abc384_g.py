def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    K = int(input[ptr])
    ptr += 1
    
    prefix_a = []
    current = []
    for i in range(N):
        current.append(A[i])
        current.sort()
        prefix_a.append(current.copy())
    
    prefix_b = []
    current = []
    for i in range(N):
        current.append(B[i])
        current.sort()
        prefix_b.append(current.copy())
    
    for _ in range(K):
        X = int(input[ptr])
        Y = int(input[ptr+1])
        ptr += 2
        a = prefix_a[X-1]
        b = prefix_b[Y-1]
        sum_total_b = sum(b)
        total = 0
        i = 0
        j = 0
        sum_b = 0
        while i < X and j < Y:
            if a[i] < b[j]:
                sum_low = sum_b
                sum_high = sum_total_b - sum_low
                cnt_low = j
                contribution = (sum_high - sum_low) + a[i] * (2 * cnt_low - Y)
                total += contribution
                i += 1
            else:
                sum_b += b[j]
                j += 1
        while i < X:
            sum_low = sum_b
            sum_high = sum_total_b - sum_low
            cnt_low = j
            contribution = (sum_high - sum_low) + a[i] * (2 * cnt_low - Y)
            total += contribution
            i += 1
        print(total)

if __name__ == '__main__':
    main()