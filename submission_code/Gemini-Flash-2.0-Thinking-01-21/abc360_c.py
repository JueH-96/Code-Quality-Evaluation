def solve():
    n = int(input())
    a = list(map(int, input().split()))
    w = list(map(int, input().split()))
    
    box_contents = [[] for _ in range(n + 1)]
    for i in range(n):
        box_contents[a[i]].append(i)
        
    total_cost = 0
    for box_index in range(1, n + 1):
        items_in_box_indices = box_contents[box_index]
        if len(items_in_box_indices) > 1:
            item_weights_indices = []
            for item_index in items_in_box_indices:
                item_weights_indices.append((w[item_index], item_index))
            item_weights_indices.sort(key=lambda pair: pair[0])
            num_to_move = len(items_in_box_indices) - 1
            for i in range(num_to_move):
                total_cost += item_weights_indices[i][0]
                
    print(total_cost)

if __name__ == '__main__':
    solve()