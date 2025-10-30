import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    sum_A = sum(A)
    if sum_A <= M:
        print("infinite")
        return
    
    A.sort()
    n = len(A)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + A[i]
    
    low = 0
    high = M
    while low < high:
        mid = (low + high + 1) // 2
        k = bisect.bisect_right(A, mid)
        total = prefix[k] + mid * (n - k)
        if total <= M:
            low = mid
        else:
            high = mid - 1
    print(low)

if __name__ == '__main__':
    main()