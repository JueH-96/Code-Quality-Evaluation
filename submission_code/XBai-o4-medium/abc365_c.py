import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    a = list(map(int, input[2:2+n]))
    sum_a = sum(a)
    if sum_a <= m:
        print("infinite")
        return
    a_sorted = sorted(a)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + a_sorted[i]
    low = 0
    high = a_sorted[-1]
    while low <= high:
        mid = (low + high) // 2
        i = bisect.bisect_left(a_sorted, mid)
        current_sum = prefix[i] + mid * (n - i)
        if current_sum <= m:
            low = mid + 1
        else:
            high = mid - 1
    print(high)

if __name__ == "__main__":
    main()