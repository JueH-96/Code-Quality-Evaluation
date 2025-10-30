# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    
    positions = {}
    for i in range(2 * N):
        if A[i] not in positions:
            positions[A[i]] = []
        positions[A[i]].append(i)
    
    count = 0
    for color in positions:
        if abs(positions[color][0] - positions[color][1]) == 2:
            count += 1
    
    print(count)

if __name__ == "__main__":
    main()