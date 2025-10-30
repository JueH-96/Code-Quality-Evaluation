import math

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            max_val = max(a[i], a[j])
            min_val = min(a[i], a[j])
            ratio = max_val / min_val
            total_sum += math.floor(ratio)
            
    print(total_sum)

if __name__ == '__main__':
    solve()