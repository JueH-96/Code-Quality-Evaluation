import bisect

def main():
    MOD = 10**8
    n = int(input())
    A = list(map(int, input().split()))
    A.sort()
    sum_total = sum(A)
    total_pairs_sum = sum_total * (n - 1)
    
    count = 0
    for i in range(n - 1):
        a = A[i]
        target = MOD - a
        j = bisect.bisect_left(A, target, i + 1, n)
        count += (n - j)
    
    result = total_pairs_sum - MOD * count
    print(result)

if __name__ == "__main__":
    main()