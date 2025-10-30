# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:]))
    
    # Create a list of available socks
    available_socks = [2] * N
    for a in A:
        available_socks[a-1] -= 1
    
    # Filter out the colors that have no socks left
    colors = [i+1 for i in range(N) if available_socks[i] > 0]
    
    # Calculate the minimum weirdness
    total_weirdness = 0
    i, j = 0, len(colors) - 1
    while i < j:
        total_weirdness += abs(colors[i] - colors[j])
        i += 1
        j -= 1
    
    print(total_weirdness)

if __name__ == "__main__":
    main()