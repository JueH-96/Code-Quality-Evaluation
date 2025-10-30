def solve():
    l, r = map(int, input().split())
    current_l = l
    result_pairs = []
    while current_l < r:
        best_len = 1
        for i in range(60, -1, -1):
            length = 1 << i
            if current_l % length == 0 and current_l + length <= r:
                best_len = length
                break
        next_r = current_l + best_len
        result_pairs.append((current_l, next_r))
        current_l = next_r
    print(len(result_pairs))
    for pair in result_pairs:
        print(pair[0], pair[1])

if __name__ == '__main__':
    solve()