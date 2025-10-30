def solve():
    n = int(input())
    a = list(map(int, input().split()))
    results = []
    for k_start in range(1, n + 1):
        current_slimes = list(a)
        takahashi_index = k_start - 1
        current_size = current_slimes[takahashi_index]
        while True:
            absorbed_in_round = False
            while True:
                absorbed_left = False
                if takahashi_index > 0 and current_slimes[takahashi_index - 1] < current_size:
                    current_size += current_slimes[takahashi_index - 1]
                    del current_slimes[takahashi_index - 1]
                    takahashi_index -= 1
                    absorbed_left = True
                    absorbed_in_round = True
                if not absorbed_left:
                    break
            while True:
                absorbed_right = False
                if takahashi_index < len(current_slimes) - 1 and current_slimes[takahashi_index + 1] < current_size:
                    current_size += current_slimes[takahashi_index + 1]
                    del current_slimes[takahashi_index + 1]
                    absorbed_right = True
                    absorbed_in_round = True
                if not absorbed_right:
                    break
            if not absorbed_in_round:
                break
        results.append(current_size)
    print(*(results))

if __name__ == '__main__':
    solve()