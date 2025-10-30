def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx + N]))
    idx += N
    B = list(map(int, input[idx:idx + N]))
    A.sort()
    B.sort()
    
    def is_possible(x):
        merged = B + [x]
        merged.sort()
        for i in range(N):
            if merged[i] < A[i]:
                return False
        return True
    
    low = 1
    high = max(A)
    answer = high
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            answer = mid
            high = mid - 1
        else:
            low = mid + 1
    print(answer)

if __name__ == '__main__':
    main()