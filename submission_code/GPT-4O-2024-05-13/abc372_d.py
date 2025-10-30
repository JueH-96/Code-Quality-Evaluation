# YOUR CODE HERE
def count_visible_buildings(N, heights):
    result = [0] * N
    stack = []
    
    for i in range(N - 1, -1, -1):
        while stack and heights[stack[-1]] < heights[i]:
            stack.pop()
        result[i] = len(stack)
        stack.append(i)
    
    return result

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
heights = list(map(int, data[1:]))

result = count_visible_buildings(N, heights)
print(" ".join(map(str, result)))