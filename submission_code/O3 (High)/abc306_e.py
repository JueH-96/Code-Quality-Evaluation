import sys
import heapq

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N = int(next(it))
    K = int(next(it))
    Q = int(next(it))

    # Current value of each position 1..N
    cur = [0] * (N + 1)
    # Whether the element is inside the current “top‐K” multiset
    in_top = [False] * (N + 1)

    # Min-heap for the current K largest elements
    top_heap = []                      # ( value ,  index )
    # Max-heap (as negatives) for the remaining positive elements
    other_heap = []                    # ( -value , -index )

    top_sum = 0                        # sum of values inside top_heap
    top_cnt = 0                        # number of VALID elements inside top_heap

    # helpers -------------------------------------------------------------
    def clean_top():
        """Remove invalid (stale) elements from the root of top_heap."""
        while top_heap:
            v, idx = top_heap[0]
            if in_top[idx] and cur[idx] == v:
                break
            heapq.heappop(top_heap)

    def clean_other():
        """Remove invalid elements from the root of other_heap."""
        while other_heap:
            nv, nidx = other_heap[0]
            v = -nv
            idx = -nidx
            if (not in_top[idx]) and cur[idx] == v and v > 0:
                break
            heapq.heappop(other_heap)
    # ---------------------------------------------------------------------

    out_lines = []

    for _ in range(Q):
        x = int(next(it))
        y = int(next(it))

        old_val = cur[x]

        # 1. delete the old value
        if in_top[x]:
            top_sum -= old_val
            top_cnt -= 1
            in_top[x] = False
        # if the old value was in 'other_heap' we just leave the stale copy

        # 2. set the new value
        cur[x] = y

        # 3. insert the new value
        if y > 0:
            if top_cnt < K:
                # Still space in Top
                heapq.heappush(top_heap, (y, x))
                top_sum += y
                in_top[x] = True
                top_cnt += 1
            else:
                clean_top()
                min_top_val = top_heap[0][0] if top_heap else None

                if min_top_val is None or y > min_top_val:
                    # Goes into Top – and the smallest Top element moves out
                    heapq.heappush(top_heap, (y, x))
                    top_sum += y
                    in_top[x] = True
                    top_cnt += 1

                    clean_top()
                    v_pop, idx_pop = heapq.heappop(top_heap)
                    top_sum -= v_pop
                    top_cnt -= 1
                    in_top[idx_pop] = False
                    if cur[idx_pop] > 0:
                        heapq.heappush(other_heap, (-v_pop, -idx_pop))
                else:
                    # Stays in Others
                    heapq.heappush(other_heap, (-y, -x))
                    # in_top[x] already False

        # 4. Refill Top if it has become too small
        while top_cnt < K:
            clean_other()
            if not other_heap:
                break
            nv, nidx = heapq.heappop(other_heap)
            v = -nv
            idx = -nidx
            heapq.heappush(top_heap, (v, idx))
            top_sum += v
            top_cnt += 1
            in_top[idx] = True

        # 5. output
        out_lines.append(str(top_sum))

    sys.stdout.write("
".join(out_lines))

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()