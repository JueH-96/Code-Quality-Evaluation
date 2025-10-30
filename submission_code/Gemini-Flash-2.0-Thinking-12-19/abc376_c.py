def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    sorted_toys = sorted(a)
    sorted_boxes_base = sorted(b)
    
    if not sorted_boxes_base:
        min_box_size = float('inf')
    else:
        min_box_size = sorted_boxes_base[0]
        
    if sorted_toys:
        min_toy_size = sorted_toys[0]
    else:
        min_toy_size = float('inf')
        
    if min_box_size < min_toy_size:
        print("-1")
        return
        
    def is_possible(x):
        current_boxes = sorted_boxes_base + [x]
        sorted_current_boxes = sorted(current_boxes)
        for i in range(n):
            if sorted_current_boxes[i] < sorted_toys[i]:
                return False
        return True
        
    low = 1
    high = 10**9 + 1
    min_x = -1
    
    while low <= high:
        mid_x = (low + high) // 2
        if is_possible(mid_x):
            min_x = mid_x
            high = mid_x - 1
        else:
            low = mid_x + 1
            
    print(min_x)

if __name__ == '__main__':
    solve()