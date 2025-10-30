# YOUR CODE HERE
import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Initialize the count of socks for each color
    count = [2] * (N + 1)
    for a in A:
        count[a] -= 1
    
    # Extract the colors that have odd counts (i.e., have one sock)
    odd_colors = []
    for i in range(1, N+1):
        if count[i] == 1:
            odd_colors.append(i)
    
    # Pair the odd colors in order to minimize the total weirdness
    total_weirdness = 0
    for i in range(0, len(odd_colors), 2):
        if i+1 < len(odd_colors):
            total_weirdness += abs(odd_colors[i] - odd_colors[i+1])
    
    print(total_weirdness)

if __name__ == "__main__":
    main()