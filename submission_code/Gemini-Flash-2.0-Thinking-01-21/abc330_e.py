def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    
    def calculate_mex(current_a):
        seen_numbers = set()
        for x in current_a:
            if x >= 0:
                seen_numbers.add(x)
        mex_val = 0
        while True:
            if mex_val not in seen_numbers:
                return mex_val
            mex_val += 1
            
    for _ in range(q):
        i_k, x_k = map(int, input().split())
        a[i_k-1] = x_k
        mex_value = calculate_mex(a)
        print(mex_value)

if __name__ == '__main__':
    solve()