import sys

def compute_sum(arr):
    arr.sort()
    res = 0
    prefix = 0
    for i, num in enumerate(arr):
        res += num * i - prefix
        prefix += num
    return res

def main():
    n = int(sys.stdin.readline())
    group0_u = []
    group0_v = []
    group1_u = []
    group1_v = []
    
    for _ in range(n):
        x, y = map(int, sys.stdin.readline().split())
        if (x - y) % 2 == 0:
            group0_u.append(x + y)
            group0_v.append(x - y)
        else:
            group1_u.append(x + y)
            group1_v.append(x - y)
    
    ans = 0
    
    sum0_u = compute_sum(group0_u)
    sum0_v = compute_sum(group0_v)
    ans += (sum0_u + sum0_v) // 2
    
    sum1_u = compute_sum(group1_u)
    sum1_v = compute_sum(group1_v)
    ans += (sum1_u + sum1_v) // 2
    
    print(ans)

if __name__ == "__main__":
    main()