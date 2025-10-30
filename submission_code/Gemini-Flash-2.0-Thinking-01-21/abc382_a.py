def solve():
    n, d = map(int, input().split())
    s = input()
    initial_empty_boxes_count = 0
    for char in s:
        if char == '.':
            initial_empty_boxes_count += 1
    final_empty_boxes_count = initial_empty_boxes_count + d
    print(final_empty_boxes_count)

if __name__ == '__main__':
    solve()