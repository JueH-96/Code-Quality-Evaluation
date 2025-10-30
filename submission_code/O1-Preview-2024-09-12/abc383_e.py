# YOUR CODE HERE
import sys
import threading
sys.setrecursionlimit(1 << 25)
def main():
    import sys
    input = sys.stdin.readline

    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((w, u - 1, v - 1))
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    A_set = set(x - 1 for x in A)
    B_set = set(x - 1 for x in B)

    parent = [i for i in range(N)]
    rank = [0] * N
    unmatched_A = [0] * N
    unmatched_B = [0] * N
    matched_pairs = [0] * N

    for x in range(N):
        if x in A_set:
            unmatched_A[x] = 1
        if x in B_set:
            unmatched_B[x] = 1

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    edges.sort()
    total_cost = 0
    for w, u, v in edges:
        ru = find(u)
        rv = find(v)
        if ru != rv:
            # Merge rv into ru
            total_unmatched_A = unmatched_A[ru] + unmatched_A[rv]
            total_unmatched_B = unmatched_B[ru] + unmatched_B[rv]
            matched_pairs_before = min(unmatched_A[ru], unmatched_B[ru]) + min(unmatched_A[rv], unmatched_B[rv])
            matched_pairs_after = min(total_unmatched_A, total_unmatched_B)
            new_pairs = matched_pairs_after - matched_pairs_before
            total_cost += new_pairs * w
            # Union by rank
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
            unmatched_A[ru] = total_unmatched_A
            unmatched_B[ru] = total_unmatched_B
            # matched_pairs[ru] = matched_pairs_after
            # (We don't need matched_pairs per se)
    print(total_cost)

threading.Thread(target=main).start()