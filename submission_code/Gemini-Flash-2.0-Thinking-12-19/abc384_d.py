def solve():
    n, s = map(int, input().split())
    a = list(map(int, input().split()))
    
    if s == 0:
        for val in a:
            if val == 0:
                print("Yes")
                return
        print("No")
        return
        
    for start_index_offset in range(n):
        current_sum = 0
        for length in range(1, 2 * n + 1):
            end_index_offset = start_index_offset + length - 1
            term_index = (end_index_offset) % n
            current_sum += a[term_index]
            if current_sum == s:
                print("Yes")
                return
            elif current_sum > s:
                break
                
    print("No")

if __name__ == '__main__':
    solve()