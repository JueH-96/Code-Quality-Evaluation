def solve():
    n = int(input())
    a = list(map(int, input().split()))
    current_a = list(a)
    operations = []
    for i in range(n):
        expected_value = i + 1
        if current_a[i] != expected_value:
            target_value = expected_value
            target_index = -1
            for j in range(i + 1, n):
                if current_a[j] == target_value:
                    target_index = j
                    break
            if target_index == -1:
                for j in range(i):
                    if current_a[j] == target_value:
                        target_index = j
                        break
                        
            if target_index != -1:
                u = min(i + 1, target_index + 1)
                v = max(i + 1, target_index + 1)
                operations.append((u, v))
                current_a[i], current_a[target_index] = current_a[target_index], current_a[i]

    print(len(operations))
    for op in operations:
        print(op[0], op[1])

if __name__ == '__main__':
    solve()