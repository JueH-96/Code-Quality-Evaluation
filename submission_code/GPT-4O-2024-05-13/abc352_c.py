# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    giants = []
    
    for i in range(N):
        A = int(data[2 * i + 1])
        B = int(data[2 * i + 2])
        giants.append((A, B))
    
    # Sort giants by A_i in ascending order
    giants.sort()
    
    # Calculate the maximum height
    max_height = 0
    current_height = 0
    
    for A, B in giants:
        current_height += A
        max_height = max(max_height, current_height + (B - A))
    
    print(max_height)

if __name__ == "__main__":
    main()