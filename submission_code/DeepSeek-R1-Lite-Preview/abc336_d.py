def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))

    def check(k):
        length = 2 * k - 1
        if length > N:
            return False
        req = list(range(1, k)) + [k] + list(range(k-1, 0, -1))
        for l in range(N - length + 1):
            valid = True
            for m in range(length):
                if A[l + m] < req[m]:
                    valid = False
                    break
            if valid:
                return True
        return False

    low = 1
    high = (N + 1) // 2
    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            low = mid + 1
        else:
            high = mid - 1
    print(high)

if __name__ == '__main__':
    main()