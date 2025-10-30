# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Initialize the result
    result = 0
    
    # Initialize the prefix XOR array
    prefix_xor = [0] * (N + 1)
    
    # Compute the prefix XOR array
    for i in range(1, N + 1):
        prefix_xor[i] = prefix_xor[i - 1] ^ A[i - 1]
    
    # Compute the result
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            result += prefix_xor[j] ^ prefix_xor[i - 1]
    
    print(result)

if __name__ == "__main__":
    solve()