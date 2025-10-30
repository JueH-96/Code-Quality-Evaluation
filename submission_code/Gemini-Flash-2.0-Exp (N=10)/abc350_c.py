def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    swaps = []
    
    for i in range(n):
        while a[i] != i + 1:
            target_index = a[i] - 1
            swaps.append((i + 1, target_index + 1))
            a[i], a[target_index] = a[target_index], a[i]
            
    print(len(swaps))
    for swap in swaps:
        print(swap[0], swap[1])

solve()