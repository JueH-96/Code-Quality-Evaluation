def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    max_xor_sum = 0
    
    def get_xor_combinations(index, current_xor, count):
        nonlocal max_xor_sum
        if index == n:
            if count == k:
                max_xor_sum = max(max_xor_sum, current_xor)
            return
        
        # Include a[index]
        get_xor_combinations(index + 1, current_xor ^ a[index], count + 1)
        # Don't include a[index]
        get_xor_combinations(index + 1, current_xor, count)
        
    get_xor_combinations(0, 0, 0)
    print(max_xor_sum)

if __name__ == '__main__':
    solve()