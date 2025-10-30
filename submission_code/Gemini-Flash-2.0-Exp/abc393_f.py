def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    
    for _ in range(q):
        r, x = map(int, input().split())
        sub_a = a[:r]
        
        def longest_increasing_subsequence(arr, max_val):
            filtered_arr = [val for val in arr if val <= max_val]
            
            if not filtered_arr:
                return 0
            
            tails = []
            
            for num in filtered_arr:
                if not tails or num > tails[-1]:
                    tails.append(num)
                else:
                    l, r = 0, len(tails) - 1
                    while l <= r:
                        mid = (l + r) // 2
                        if tails[mid] < num:
                            l = mid + 1
                        else:
                            r = mid - 1
                    tails[l] = num
            
            return len(tails)
        
        print(longest_increasing_subsequence(sub_a, x))

solve()