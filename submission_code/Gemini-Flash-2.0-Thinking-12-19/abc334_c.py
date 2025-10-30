def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    lost_colors = set(a)
    sock_counts = {}
    for color in range(1, n + 1):
        sock_counts[color] = 2
    for color in lost_colors:
        sock_counts[color] -= 1
    
    same_color_pairs_weirdness = 0
    remaining_sock_colors = []
    for color in range(1, n + 1):
        same_color_pairs_count = sock_counts[color] // 2
        same_color_pairs_weirdness += 0 * same_color_pairs_count
        sock_counts[color] %= 2
        if sock_counts[color] == 1:
            remaining_sock_colors.append(color)
            
    remaining_socks_count = len(remaining_sock_colors)
    if remaining_socks_count % 2 == 0:
        weirdness = 0
        sorted_colors = sorted(remaining_sock_colors)
        for i in range(0, remaining_socks_count, 2):
            weirdness += abs(sorted_colors[i] - sorted_colors[i+1])
        print(weirdness)
    else:
        min_weirdness = float('inf')
        sorted_colors = sorted(remaining_sock_colors)
        for i in range(remaining_socks_count):
            current_colors = []
            for j in range(remaining_socks_count):
                if i != j:
                    current_colors.append(sorted_colors[j])
            current_weirdness = 0
            for j in range(0, len(current_colors), 2):
                current_weirdness += abs(current_colors[j] - current_colors[j+1])
            min_weirdness = min(min_weirdness, current_weirdness)
        print(min_weirdness)

if __name__ == '__main__':
    solve()