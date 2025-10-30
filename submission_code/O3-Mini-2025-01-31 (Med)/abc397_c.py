def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    arr = list(map(int, data[1:]))
    
    # Compute left distinct counts:
    # left[i] holds the count of distinct integers in arr[0 ... i]
    left = [0] * n
    seen = set()
    count = 0
    for i in range(n):
        if arr[i] not in seen:
            count += 1
            seen.add(arr[i])
        left[i] = count

    # Compute right distinct counts:
    # right[i] holds the count of distinct integers in arr[i ... n-1]
    right = [0] * n
    seen.clear()
    count = 0
    for i in range(n-1, -1, -1):
        if arr[i] not in seen:
            count += 1
            seen.add(arr[i])
        right[i] = count

    # Try splitting at every possible position and record the max sum.
    max_sum = 0
    for i in range(0, n - 1):  # split between i and i+1
        total = left[i] + right[i + 1]
        if total > max_sum:
            max_sum = total

    sys.stdout.write(str(max_sum))


if __name__ == '__main__':
    main()