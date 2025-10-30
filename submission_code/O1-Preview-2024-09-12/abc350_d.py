# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math

    import sys

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M = map(int, sys.stdin.readline().split())
    neighbors = [[] for _ in range(N+1)]
    neighbor_set = [set() for _ in range(N+1)]
    for _ in range(M):
        A_i, B_i = map(int, sys.stdin.readline().split())
        neighbors[A_i].append(B_i)
        neighbors[B_i].append(A_i)
        neighbor_set[A_i].add(B_i)
        neighbor_set[B_i].add(A_i)

    total_operations = 0

    for Y in range(1, N+1):
        friends_Y = neighbors[Y]
        deg_Y = len(friends_Y)
        if deg_Y >= 2:
            friends_Y_sorted = sorted(friends_Y)
            for i in range(len(friends_Y)):
                u = friends_Y_sorted[i]
                for j in range(i+1, len(friends_Y)):
                    v = friends_Y_sorted[j]
                    if v not in neighbor_set[u]:
                        total_operations +=1

    print(total_operations)



if __name__ == "__main__":
    threading.Thread(target=main).start()