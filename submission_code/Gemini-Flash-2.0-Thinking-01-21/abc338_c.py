import math

def solve():
    n = int(input())
    q = list(map(int, input().split()))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    def is_possible(s):
        lower_bound = 0
        upper_bound = s
        for i in range(n):
            c_i = a[i] - b[i]
            d_i = q[i] - b[i] * s
            if c_i > 0:
                upper_bound = min(upper_bound, d_i // c_i if c_i != 0 else float('inf'))
            elif c_i < 0:
                lower_bound = max(lower_bound, math.ceil(d_i / c_i) if c_i != 0 else float('-inf'))
            elif c_i == 0:
                if d_i < 0:
                    return False
        if lower_bound <= upper_bound and lower_bound <= s and upper_bound >= 0:
            return True
        else:
            return False
            
    max_servings = 0
    low = 0
    high = sum(q) + 1 # A sufficiently large upper bound
    
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            max_servings = mid
            low = mid + 1
        else:
            high = mid - 1
            
    print(max_servings)

if __name__ == '__main__':
    solve()