from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    H = list(map(int, input[1:n+1]))
    
    positions = defaultdict(list)
    for idx, h in enumerate(H):
        positions[h].append(idx)
    
    max_buildings = 1  # At least one building can always be chosen
    
    for h, pos_list in positions.items():
        m = len(pos_list)
        if m < 2:
            continue
        # Initialize a list of dictionaries for dynamic programming
        dp = [{} for _ in range(m)]
        current_max = 1
        for k in range(1, m):
            for j in range(k):
                d = pos_list[k] - pos_list[j]
                # Get the previous length or default to 1 (only the current element)
                prev_length = dp[j].get(d, 1)
                new_length = prev_length + 1
                dp[k][d] = new_length
                if new_length > current_max:
                    current_max = new_length
        if current_max > max_buildings:
            max_buildings = current_max
    
    print(max_buildings)

if __name__ == "__main__":
    main()