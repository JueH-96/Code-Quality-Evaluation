def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+K]))
    
    # Initialize the count of socks for each color
    count = [2] * N
    for a in A:
        count[a-1] -= 1
    
    # Create a list of colors with odd counts
    odd_colors = []
    for i in range(N):
        if count[i] == 1:
            odd_colors.append(i+1)
    
    # Pair the odd colors to minimize the total weirdness
    total_weirdness = 0
    for i in range(0, len(odd_colors)-1, 2):
        total_weirdness += abs(odd_colors[i] - odd_colors[i+1])
    
    print(total_weirdness)

if __name__ == "__main__":
    main()