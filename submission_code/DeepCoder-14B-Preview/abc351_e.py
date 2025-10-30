def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    points = []
    idx = 1
    for _ in range(N):
        x = int(data[idx])
        y = int(data[idx + 1])
        points.append((x, y))
        idx += 2
    
    even = []
    odd = []
    for x, y in points:
        s = x + y
        if s % 2 == 0:
            even.append((x, y))
        else:
            odd.append((x, y))
    
    def sum_abs(arr):
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + arr_sorted[i]
        total = 0
        for i in range(n):
            total += arr_sorted[i] * i - prefix[i]
        return total
    
    def compute_group(group):
        n = len(group)
        if n < 2:
            return 0
        u = [x + y for x, y in group]
        v = [x - y for x, y in group]
        sum_u = sum_abs(u)
        sum_v = sum_abs(v)
        return (sum_u + sum_v) // 2
    
    total = compute_group(even) + compute_group(odd)
    print(total)

if __name__ == "__main__":
    main()