import sys

# Read input
N = int(sys.stdin.readline().strip())
heights = [int(x) for x in sys.stdin.readline().strip().split()]

# Find the maximum number of buildings that can be chosen
def max_buildings():
    max_count = 1
    for i in range(N):
        for j in range(i+1, N):
            if heights[j] == heights[i]:
                count = 2
                step = j - i
                for k in range(j+step, N, step):
                    if k < N and heights[k] == heights[i]:
                        count += 1
                    else:
                        break
                max_count = max(max_count, count)
    return max_count

print(max_buildings())