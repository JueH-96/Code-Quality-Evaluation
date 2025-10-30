def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    count = 0
    
    def find_subsequences(a, b, a_idx, b_idx, current_subsequence):
        nonlocal count
        if b_idx == len(b):
            count += 1
            return
        
        if a_idx == len(a):
            return

        if a[a_idx] == b[b_idx]:
            find_subsequences(a, b, a_idx + 1, b_idx + 1, current_subsequence + [a[a_idx]])
            
        find_subsequences(a, b, a_idx + 1, b_idx, current_subsequence)

    find_subsequences(a, b, 0, 0, [])

    if count >= 2:
        print("Yes")
    else:
        print("No")

solve()