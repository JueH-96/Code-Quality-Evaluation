def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    min_toy_size = min(a)
    possible = True
    for box_size in b:
        if min_toy_size > box_size:
            possible = False
            break
            
    if not possible:
        print("-1")
        return
        
    min_x = -1
    low = 1
    high = 10**9 + 1
    
    while low < high:
        mid_x = (low + high) // 2
        current_boxes = b + [mid_x]
        sorted_toy_sizes = sorted(a)
        sorted_box_sizes = sorted(current_boxes)
        valid_x = True
        for i in range(n):
            if sorted_toy_sizes[i] > sorted_box_sizes[i]:
                valid_x = False
                break
        if valid_x:
            min_x = mid_x
            high = mid_x
        else:
            low = mid_x + 1
            
    print(min_x)

if __name__ == '__main__':
    solve()