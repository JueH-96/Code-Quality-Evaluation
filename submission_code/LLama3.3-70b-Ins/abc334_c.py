import sys

def calculate_weirdness(N, K, A):
    # Initialize a list to store the count of socks for each color
    sock_count = [2] * (N + 1)
    
    # Decrease the count of socks for each lost color
    for color in A:
        sock_count[color] -= 1
    
    # Initialize variables to store the total weirdness and the number of pairs
    total_weirdness = 0
    num_pairs = (2 * N - K) // 2
    
    # Initialize two pointers, one at the start and one at the end of the sock count list
    left = 1
    right = N
    
    # Try to form pairs with the same color
    for color in range(1, N + 1):
        if sock_count[color] >= 2:
            pairs = sock_count[color] // 2
            num_pairs -= pairs
            sock_count[color] -= pairs * 2
    
    # Try to form pairs with different colors
    while num_pairs > 0 and left < right:
        if sock_count[left] > 0 and sock_count[right] > 0:
            total_weirdness += right - left
            sock_count[left] -= 1
            sock_count[right] -= 1
            num_pairs -= 1
        elif sock_count[left] > 0:
            left += 1
        else:
            right -= 1
    
    return total_weirdness

def main():
    # Read the input from stdin
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Calculate and print the minimum total weirdness
    print(calculate_weirdness(N, K, A))

if __name__ == "__main__":
    main()