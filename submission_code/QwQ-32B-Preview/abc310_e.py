def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    A = [int(c) for c in S]
    
    sum_prev = 0
    total_sum = 0
    for j in range(1, N + 1):
        if A[j - 1] == 0:
            sum_j = j - 1
        else:
            sum_j = j - sum_prev
        total_sum += sum_j
        sum_prev = sum_j
    print(total_sum)

if __name__ == "__main__":
    main()