import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = int(next(it))
    B = int(next(it))
    D = [int(next(it)) for _ in range(N)]

    L = A + B

    # Arc representation on the circle [0..L-1]:
    #   - If arc["empty"] == True: empty arc.
    #   - Else if arc["full"] == True: the entire circle.
    #   - Else: arc from arc["s"] to arc["e"] (inclusive) going forward mod L.
    def make_full():
        return {"empty": False, "full": True, "s": 0, "e": L-1}

    def make_empty():
        return {"empty": True, "full": False, "s": 0, "e": -1}

    def make_arc(s, e):
        # Create an arc from s to e inclusive (mod L).  s,e in [0..L-1]
        return {"empty": False, "full": False, "s": s, "e": e}

    def arc_to_list(arc):
        # Convert an arc into a list of linear intervals on [0..L-1].
        if arc["empty"]:
            return []
        if arc["full"]:
            return [(0, L-1)]
        s, e = arc["s"], arc["e"]
        if s <= e:
            return [(s, e)]
        else:
            # wrap-around
            return [(s, L-1), (0, e)]

    def intersect_arcs(a, b):
        # Intersection of two arcs a and b, each in our representation.
        if a["empty"] or b["empty"]:
            return make_empty()
        if a["full"]:
            return b.copy()
        if b["full"]:
            return a.copy()

        L1 = arc_to_list(a)
        L2 = arc_to_list(b)
        segs = []
        for (l1, r1) in L1:
            for (l2, r2) in L2:
                l = max(l1, l2)
                r = min(r1, r2)
                if l <= r:
                    segs.append((l, r))
        if not segs:
            return make_empty()

        # segs length is at most 2.  Build single arc representation.
        segs.sort()
        if len(segs) == 1:
            l, r = segs[0]
            # check if it's full
            if l == 0 and r == L-1:
                return make_full()
            else:
                return make_arc(l, r)
        else:
            # two segments: must be wrap-around
            # segs[0] = (0, r0), segs[1] = (l1, L-1)
            (l0, r0), (l1, r1) = segs
            # They should align to wrap
            if l0 == 0 and r1 == L-1:
                # wrap arc from l1 to r0
                if l1 == 0 and r0 == L-1:
                    return make_full()
                return make_arc(l1, r0)
            else:
                # no valid wrap: intersection would be two disjoint pieces which
                # can't arise from intersecting two connected arcs on a circle.
                # But just in case, treat as empty.
                return make_empty()

    # Start with full circle
    cur = make_full()

    for d in D:
        r = d % L
        # solve x + r ≡ y  with y in [0..A-1]  =>  x ≡ y - r  so x in [-r .. A-1-r] mod L
        # shift into [0..L-1]:
        s = (-r) % L
        e = (s + A - 1) % L
        constraint = make_arc(s, e)
        cur = intersect_arcs(cur, constraint)
        if cur["empty"]:
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    main()