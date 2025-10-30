def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    sum_A = sum(A)
    low = 0
    high = sum_A
    best_x = 0

    def is_possible(x):
        if sum_A < K * x:
            return False
        current_sum = 0
        count = 0
        splits = []
        for i in range(N):
            current_sum += A[i]
            if current_sum >= x:
                count += 1
                current_sum = 0
                splits.append(i)
                if count >= K:
                    break
        return count >= K

    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            best_x = mid
            low = mid + 1
        else:
            high = mid - 1

    # Calculate the number of cut lines that are never cut
    num_splits = len(splits) if 'splits' in locals() else 0
    result = (N - 1) - num_splits

    print(best_x, result)

if __name__ == '__main__':
    main()