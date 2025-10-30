def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+K]))
    
    # Step 1: Calculate the number of remaining socks for each color
    remaining_socks = [2] * (N + 1)  # index 0 is unused, colors are 1 to N
    for a in A:
        remaining_socks[a] -= 1
    
    # Step 2: Create a list of remaining socks
    remaining_list = []
    for color in range(1, N + 1):
        for _ in range(remaining_socks[color]):
            remaining_list.append(color)
    
    # Step 3: Sort the list of remaining socks
    remaining_list.sort()
    
    # Step 4: Pair the socks and calculate the total weirdness
    total_weirdness = 0
    for i in range(0, len(remaining_list) - 1, 2):
        total_weirdness += abs(remaining_list[i] - remaining_list[i + 1])
    
    # Output the result
    print(total_weirdness)