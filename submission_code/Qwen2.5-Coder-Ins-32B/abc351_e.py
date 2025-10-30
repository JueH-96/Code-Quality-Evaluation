# YOUR CODE HERE
import sys
import collections

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    points = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(N)]

    def dist(p1, p2):
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        if (dx + dy) % 2 != 0:
            return 0
        return (dx + dy) // 2

    sum_dist = 0
    for i in range(N):
        for j in range(i + 1, N):
            sum_dist += dist(points[i], points[j])

    print(sum_dist)

if __name__ == "__main__":
    main()