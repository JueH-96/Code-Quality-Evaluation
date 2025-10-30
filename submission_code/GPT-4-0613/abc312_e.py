import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    cuboids = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    cuboids.sort()

    cuboid_dict = defaultdict(list)
    for i in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[i]
        cuboid_dict[(x1, y1, z1)].append(i)
        cuboid_dict[(x2, y2, z2)].append(i)

    adj = [set() for _ in range(N)]
    for cuboid_list in cuboid_dict.values():
        for i in range(len(cuboid_list)):
            for j in range(i+1, len(cuboid_list)):
                adj[cuboid_list[i]].add(cuboid_list[j])
                adj[cuboid_list[j]].add(cuboid_list[i])

    for i in range(N):
        x1, y1, z1, x2, y2, z2 = cuboids[i]
        for j in cuboid_dict[(x1, y1, z1)]:
            if j != i:
                adj[i].remove(j)
        for j in cuboid_dict[(x2, y2, z2)]:
            if j != i:
                adj[i].remove(j)

    for i in range(N):
        print(len(adj[i]))

if __name__ == "__main__":
    main()