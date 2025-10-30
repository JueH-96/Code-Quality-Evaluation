def solve():
    n = int(input())
    a = list(map(int, input().split()))
    if n <= 1:
        print(0)
        return
    
    s = sum(a)
    x = s // n
    lower_bound = x
    upper_bound = x + 1
    
    decrease_needed = 0
    increase_needed = 0
    
    for val in a:
        if val > upper_bound:
            decrease_needed += val - upper_bound
        elif val < lower_bound:
            increase_needed += lower_bound - val
            
    print(max(decrease_needed, increase_needed))

if __name__ == '__main__':
    solve()