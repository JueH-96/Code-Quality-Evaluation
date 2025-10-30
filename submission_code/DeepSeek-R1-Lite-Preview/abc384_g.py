def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx + N]))
    idx += N
    B = list(map(int, data[idx:idx + N]))
    idx += N
    K = int(data[idx])
    idx += 1
    queries = []
    for _ in range(K):
        X = int(data[idx])
        Y = int(data[idx + 1])
        queries.append((X, Y))
        idx += 2
    
    A_sorted = sorted(A)
    B_sorted = sorted(B)
    
    prefix_sum_A = [0] * (N + 1)
    for i in range(N):
        prefix_sum_A[i + 1] = prefix_sum_A[i] + A_sorted[i]
    
    prefix_sum_B = [0] * (N + 1)
    for i in range(N):
        prefix_sum_B[i + 1] = prefix_sum_B[i] + B_sorted[i]
    
    for X, Y in queries:
        sum_total = 0
        i = 0
        j = 0
        sum_a = 0
        cnt_a = 0
        sum_b = 0
        while i < X and j < Y:
            if A_sorted[i] < B_sorted[j]:
                sum_total += B_sorted[j] * cnt_a - sum_a
                sum_b += B_sorted[j]
                j += 1
            else:
                sum_total += A_sorted[i] * j - sum_b
                sum_a += A_sorted[i]
                cnt_a += 1
                i += 1
        if i < X:
            sum_total += (prefix_sum_A[X] - sum_a) * Y - sum_b * (X - i)
        if j < Y:
            sum_total += sum_b * X - (prefix_sum_B[Y] - sum_b) * X
        print(sum_total)

if __name__ == '__main__':
    main()