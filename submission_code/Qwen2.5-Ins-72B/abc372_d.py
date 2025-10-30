# YOUR CODE HERE
import sys

def solve(n, heights):
    stack = []
    result = [0] * n
    
    for i in range(n - 1, -1, -1):
        while stack and heights[i] > heights[stack[-1]]:
            stack.pop()
        result[i] = len(stack)
        stack.append(i)
    
    return result

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    heights = list(map(int, data[1:]))
    result = solve(n, heights)
    print(" ".join(map(str, result)))