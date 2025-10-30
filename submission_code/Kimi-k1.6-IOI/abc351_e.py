def sum_abs_diff(arr):
    arr.sort()
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + arr[i]
    total = 0
    for i in range(n):
        total += arr[i] * i - prefix[i]
    return total

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    group0_u = []
    group0_v = []
    group1_u = []
    group1_v = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx + 1])
        idx += 2
        u = x + y
        v = x - y
        if u % 2 == 0:
            group0_u.append(u)
            group0_v.append(v)
        else:
            group1_u.append(u)
            group1_v.append(v)
    sum_u = sum_abs_diff(group0_u) + sum_abs_diff(group1_u)
    sum_v = sum_abs_diff(group0_v) + sum_abs_diff(group1_v)
    print((sum_u + sum_v) // 2)

if __name__ == "__main__":
    main()