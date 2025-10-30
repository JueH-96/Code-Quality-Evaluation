def count_visible_buildings(N, heights):
    result = [0] * N
    stack = []

    for i in range(N):
        while stack and heights[stack[-1]] < heights[i]:
            j = stack.pop()
            result[j] += 1
            result[j] += len(stack)  # All buildings in the stack are visible to j
        stack.append(i)

    return result

import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
heights = list(map(int, data[1:N+1]))

result = count_visible_buildings(N, heights)
print(' '.join(map(str, result)))