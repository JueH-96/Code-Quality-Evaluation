def sum_abs_diffs(arr):
    arr.sort()
    res = 0
    prefix = 0
    for i in range(len(arr)):
        res += arr[i] * i - prefix
        prefix += arr[i]
    return res

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    idx = 1
    even = []
    odd = []
    for _ in range(n):
        x = int(input[idx])
        y = int(input[idx+1])
        idx += 2
        if (x + y) % 2 == 0:
            even.append((x, y))
        else:
            odd.append((x, y))
    
    total = 0
    for group in [even, odd]:
        m = len(group)
        if m < 2:
            continue
        u = [x + y for x, y in group]
        v = [x - y for x, y in group]
        sum_u = sum_abs_diffs(u)
        sum_v = sum_abs_diffs(v)
        total += (sum_u + sum_v) // 2
    print(total)

if __name__ == "__main__":
    main()