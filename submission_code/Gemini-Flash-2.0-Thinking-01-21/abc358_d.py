def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    sorted_requirements = sorted(b)
    boxes_with_indices = []
    for i in range(n):
        boxes_with_indices.append({'candies': a[i], 'price': a[i], 'index': i+1})
    
    sorted_boxes = sorted(boxes_with_indices, key=lambda x: x['candies'])
    
    used_box_indices = set()
    total_cost = 0
    
    for requirement in sorted_requirements:
        found_box = False
        for box_info in sorted_boxes:
            if box_info['candies'] >= requirement and box_info['index'] not in used_box_indices:
                total_cost += box_info['price']
                used_box_indices.add(box_info['index'])
                found_box = True
                break
        if not found_box:
            print("-1")
            return
            
    print(total_cost)

if __name__ == '__main__':
    solve()