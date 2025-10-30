def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    sock_counts = {}
    for i in range(1, n + 1):
        sock_counts[i] = 2
        
    for color in a:
        sock_counts[color] -= 1
        
    remaining_socks_colors = []
    for color in range(1, n + 1):
        for _ in range(sock_counts[color]):
            remaining_socks_colors.append(color)
            
    pairs_to_make = (2 * n - k) // 2
    same_color_pairs = 0
    weirdness = 0
    
    color_counts = {}
    for color in range(1, n + 1):
        color_counts[color] = sock_counts[color]
        
    for color in range(1, n + 1):
        pairs = color_counts[color] // 2
        same_color_pairs += pairs
        color_counts[color] %= 2
        
    remaining_colors_for_pairing = []
    for color in range(1, n + 1):
        if color_counts[color] > 0:
            remaining_colors_for_pairing.append(color)
            
    remaining_colors_for_pairing.sort()
    
    additional_pairs_weirdness = 0
    num_additional_pairs = len(remaining_colors_for_pairing) // 2
    
    for i in range(0, len(remaining_colors_for_pairing), 2):
        if i + 1 < len(remaining_colors_for_pairing):
            additional_pairs_weirdness += abs(remaining_colors_for_pairing[i] - remaining_colors_for_pairing[i+1])
            
    print(additional_pairs_weirdness)

if __name__ == '__main__':
    solve()