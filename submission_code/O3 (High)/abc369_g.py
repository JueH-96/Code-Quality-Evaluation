import sys
import threading
import sys, math, bisect, heapq

# small recursion depth fix
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    adj = [[] for _ in range(N+1)]
    total_weight = 0
    for _ in range(N-1):
        u,v,w = map(int, input().split())
        adj[u].append((v,w))
        adj[v].append((u,w))
        total_weight += w

    # DFS returning a max-heap (as negative values for min-heap) that keeps all “increment” values
    def dfs(v, parent, in_w):
        # heap is a list functioning as min-heap of NEGATIVE numbers
        heap = []
        for to, w in adj[v]:
            if to == parent:
                continue
            child_heap = dfs(to, v, w)              # heap from child
            # merge small to large
            if len(child_heap) > len(heap):
                heap, child_heap = child_heap, heap
            for val in child_heap:
                heapq.heappush(heap, val)

        if not heap:                      # leaf  (no child heaps merged)
            if in_w:                      # root has in_w==0, never goes here
                heapq.heappush(heap, -in_w)
        else:
            if in_w:                      # add in_w to current largest element
                largest = -heapq.heappop(heap)
                largest += in_w
                heapq.heappush(heap, -largest)
        return heap

    root_heap = dfs(1, 0, 0)                 # heap of all increment values
    incs = sorted((-x for x in root_heap), reverse=True)   # convert to descending positive list

    # prefix sums of increments
    prefix = [0]
    for val in incs:
        prefix.append(prefix[-1] + val)

    max_full = total_weight * 2              # full tour length when every edge is covered

    out_lines = []
    leaves = len(incs)
    for k in range(1, N+1):
        if k <= leaves:
            score = prefix[k] * 2
        else:
            score = max_full
        out_lines.append(str(score))
    sys.stdout.write("
".join(out_lines))

# make it faster for CP constraints
if __name__ == "__main__":
    threading.Thread(target=main).start()