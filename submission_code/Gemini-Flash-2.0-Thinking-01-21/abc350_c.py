def solve():
    n = int(input())
    a = list(map(int, input().split()))
    operations = []
    pos_map = {}
    for i in range(n):
        pos_map[a[i]] = i
    
    for k in range(n - 1):
        target_value = k + 1
        target_position = k
        current_position = pos_map[target_value]
        if current_position != target_position:
            operations.append((target_position + 1, current_position + 1))
            value_at_k = a[target_position]
            value_at_current_pos = a[current_position]
            a[target_position] = value_at_current_pos
            a[current_position] = value_at_k
            pos_map[value_at_k] = current_position
            pos_map[value_at_current_pos] = target_position
            
    print(len(operations))
    for op in operations:
        print(op[0], op[1])

if __name__ == '__main__':
    solve()