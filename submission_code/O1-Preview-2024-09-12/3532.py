class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        import sys
        import threading
        sys.setrecursionlimit(1 << 25)
        def main():
            n = len(edges) +1
            tree = [[] for _ in range(n)]
            for u, v in edges:
                tree[u].append(v)
                tree[v].append(u)
            # First, compute adjusted_depth_1[u] in post-order traversal
            adjusted_depth_1 = [0]*n  # Max adjusted depth in subtree rooted at u
            adjusted_max = [0]*n  # Max adjusted distance from u to any node
            def dfs1(u, parent):
                # Compute adjusted_depth_1[u]
                max_depth = 0
                for v in tree[u]:
                    if v != parent:
                        dfs1(v, u)
                        delay_v = 1 if v%2==1 else 2  # delay(v)
                        depth = delay_v + adjusted_depth_1[v]
                        if depth > max_depth:
                            max_depth = depth
                adjusted_depth_1[u] = max_depth

            dfs1(0, -1)
            # Now compute adjusted_max[u] for all nodes
            def dfs2(u, parent, adjusted_from_parent):
                # adjusted_max[u] = max(adjusted_depth_1[u], adjusted_from_parent)
                adjusted_max[u] = max(adjusted_depth_1[u], adjusted_from_parent)
                # Prepare list of children's adjusted depths with delays
                ch = []
                for v in tree[u]:
                    if v != parent:
                        delay_v = 1 if v%2==1 else 2
                        ch.append((delay_v + adjusted_depth_1[v], v))
                # Find the two largest depths among children
                first_max, second_max = 0, 0
                first_v, second_v = -1, -1
                for depth, v in ch:
                    if depth > first_max:
                        second_max = first_max
                        second_v = first_v
                        first_max = depth
                        first_v = v
                    elif depth > second_max:
                        second_max = depth
                        second_v = v
                for depth, v in ch:
                    if v == first_v:
                        use_depth = second_max
                    else:
                        use_depth = first_max
                    delay_u = 1 if u%2==1 else 2
                    # adjusted_from_parent_v = max(adjusted_from_parent + delay(u), use_depth) + delay(u)
                    adjusted_from_parent_v = max(adjusted_from_parent + delay_u, use_depth) + delay_u
                    dfs2(v, u, adjusted_from_parent_v)
            dfs2(0, -1, 0)
            # Now, times[i] = adjusted_max[i]
            times = adjusted_max
            return times
        threading.Thread(target=main).start()