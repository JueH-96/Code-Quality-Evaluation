import collections
import sys

def main():
    N, M, K = map(int, sys.stdin.readline().split())

    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # max_stamina[i] stores the maximum remaining stamina a guard would have upon reaching vertex i
    # Initialized to -1 (meaning not reachable effectively yet)
    max_stamina = [-1] * (N + 1)
    
    # buckets[s] will store a deque of vertices that can be reached with exactly s stamina remaining
    # Max possible stamina h_i is N. So buckets for staminas 0 to N.
    buckets = [collections.deque() for _ in range(N + 1)]

    # Initialize guard locations
    for _ in range(K):
        p, h = map(int, sys.stdin.readline().split())
        # A guard is at p with stamina h.
        # Since max_stamina[p] is initially -1 and h >= 1 (problem constraint),
        # h > max_stamina[p] is always true for the first assignment for a guard at p.
        # This correctly sets max_stamina[p] = h and adds p to bucket[h].
        if h > max_stamina[p]: 
            max_stamina[p] = h
            buckets[h].append(p)

    # Process buckets from highest stamina down to 0. This is a Dijkstra-like approach.
    # s is the current stamina level being processed.
    for s in range(N, -1, -1): # s from N down to 0
        queue_at_this_stamina_level = buckets[s]
        while queue_at_this_stamina_level:
            u = queue_at_this_stamina_level.popleft()

            # If max_stamina[u] was improved after u was added to this bucket (buckets[s]),
            # it means u was (or will be) processed from a higher stamina bucket (s' > s).
            # So, this entry (u from buckets[s]) is stale/suboptimal. Skip.
            if max_stamina[u] > s:
                continue
            
            # If current stamina is 0, u is guarded, but cannot guard its neighbors further
            # as stamina would become -1.
            if s == 0:
                continue

            # Explore neighbors: they will be reached with stamina s-1
            next_s = s - 1
            for v in adj[u]:
                # If this path offers more stamina to v than previously known
                if next_s > max_stamina[v]:
                    max_stamina[v] = next_s
                    buckets[next_s].append(v) # Add v to be processed at its new stamina level

    guarded_vertices = []
    for i in range(1, N + 1):
        if max_stamina[i] >= 0: # Any vertex with non-negative remaining stamina is guarded
            guarded_vertices.append(i)
    
    # guarded_vertices are collected by iterating i from 1 to N, so they are already sorted.

    sys.stdout.write(str(len(guarded_vertices)) + "
")
    # K >= 1 and h_i >= 1, so there is always at least one guarded vertex.
    # Thus, guarded_vertices list is never empty.
    sys.stdout.write(" ".join(map(str, guarded_vertices)) + "
")

if __name__ == '__main__':
    main()