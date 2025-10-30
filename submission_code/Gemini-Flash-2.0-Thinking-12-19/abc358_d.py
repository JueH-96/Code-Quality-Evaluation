def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    boxes = []
    for i in range(n):
        boxes.append({'candy_count': a[i], 'original_index': i+1})
    
    boxes.sort(key=lambda x: x['candy_count'])
    
    sorted_b = sorted(b)
    used_indices = set()
    total_cost = 0
    
    for req in sorted_b:
        found_box = False
        for box in boxes:
            if box['candy_count'] >= req and box['original_index'] not in used_indices:
                total_cost += box['candy_count']
                used_indices.add(box['original_index'])
                found_box = True
                break
        if not found_box:
            print("-1")
            return
            
    print(total_cost)

if __name__ == '__main__':
    solve()