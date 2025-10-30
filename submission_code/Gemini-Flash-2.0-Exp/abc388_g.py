def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())
    
    for _ in range(q):
        l, r = map(int, input().split())
        l -= 1
        
        sub_array = a[l:r]
        
        count = 0
        used = [False] * len(sub_array)
        
        for i in range(len(sub_array)):
            if not used[i]:
                for j in range(len(sub_array) - 1, i, -1):
                    if not used[j] and sub_array[i] * 2 <= sub_array[j]:
                        count += 1
                        used[i] = True
                        used[j] = True
                        break
        print(count)

solve()