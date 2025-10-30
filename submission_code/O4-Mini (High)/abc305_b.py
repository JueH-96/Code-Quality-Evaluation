def main():
    p, q = input().split()
    # distances between adjacent points A-B, B-C, C-D, D-E, E-F, F-G
    distances = [3, 1, 4, 1, 5, 9]
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # compute prefix sums to get coordinate of each point
    pos = {}
    curr = 0
    for i, label in enumerate(labels):
        pos[label] = curr
        if i < len(distances):
            curr += distances[i]
    # answer is the absolute difference of their coordinates
    print(abs(pos[p] - pos[q]))

main()