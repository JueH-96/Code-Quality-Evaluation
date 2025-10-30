import math

def solve():
    n = int(input())
    buildings = []
    for _ in range(n):
        x, h = map(int, input().split())
        buildings.append((x, h))

    def can_see(observer_h, target_building_index, all_buildings):
        ox = 0
        oy = observer_h
        tx, th = all_buildings[target_building_index]

        for other_index, (ox2, oh2) in enumerate(all_buildings):
            if other_index == target_building_index:
                continue

            for ty in [0, oh2]:
                # Check if the line segment from observer to top/bottom of other building blocks the view
                p1 = (ox, oy)
                p2 = (ox2, ty)

                # Check if the line segment p1-p2 intersects the target building
                def intersects(p1, p2, rect_x, rect_h):
                    def on_segment(p, a, b):
                        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]) and
                                (p[0] - a[0]) * (b[1] - a[1]) == (p[1] - a[1]) * (b[0] - a[0]))

                    def line_segment_intersection(a, b, c, d):
                        def orientation(p, q, r):
                            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                            if val == 0: return 0  # Collinear
                            return 1 if val > 0 else 2  # Clockwise or Counterclockwise

                        o1 = orientation(a, b, c)
                        o2 = orientation(a, b, d)
                        o3 = orientation(c, d, a)
                        o4 = orientation(c, d, b)

                        if o1 != o2 and o3 != o4:
                            return True

                        if o1 == 0 and on_segment(c, a, b): return True
                        if o2 == 0 and on_segment(d, a, b): return True
                        if o3 == 0 and on_segment(a, c, d): return True
                        if o4 == 0 and on_segment(b, c, d): return True

                        return False

                    return line_segment_intersection(p1, p2, (rect_x, 0), (rect_x, rect_h))

                if intersects(p1, p2, tx, th):
                    return False
        return True

    def can_see_all(observer_h, all_buildings):
        for i in range(len(all_buildings)):
            is_visible = False
            tx, th = all_buildings[i]
            for ty in [0 + 1e-9, th - 1e-9]: # Check top and bottom visibility
                can_reach_top_or_bottom = True
                for other_j, (ox2, oh2) in enumerate(all_buildings):
                    if i == other_j:
                        continue

                    def intersects(p1, p2, rect_x, rect_h):
                        # Simplified intersection check
                        if (p1[0] <= rect_x <= p2[0] or p2[0] <= rect_x <= p1[0]):
                            if p1[0] == p2[0]: return False # Vertical line, handled by x check
                            y_at_rect = p1[1] + (p2[1] - p1[1]) * (rect_x - p1[0]) / (p2[0] - p1[0])
                            return 0 <= y_at_rect <= rect_h
                        return False

                    if intersects((0, observer_h), (tx, ty), ox2, oh2):
                        can_reach_top_or_bottom = False
                        break
                if can_reach_top_or_bottom:
                    is_visible = True
                    break
            if not is_visible:
                return False
        return True

    if can_see_all(0, buildings):
        print(-1)
        return

    critical_heights = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x1, h1 = buildings[i]
            x2, h2 = buildings[j]

            def calculate_h(h_target, x_target, y_target, x_other, y_other_building):
                if x_target - 0 == 0: return None
                if x_other - 0 == 0: return None

                if x_target == x_other: return None

                h = (y_target * x_other - y_other_building * x_target) / (x_other - x_target)
                return h

            for y1 in [0, h1]:
                for y2 in [0, h2]:
                    h_val = calculate_h(0, x1, y1, x2, y2)
                    if h_val is not None and h_val >= 0:
                        critical_heights.add(h_val)

    sorted_heights = sorted(list(critical_heights))

    max_h_not_all_visible = 0.0
    epsilon = 1e-10

    for h in sorted_heights:
        if not can_see_all(h - epsilon, buildings):
            max_h_not_all_visible = max(max_h_not_all_visible, h - epsilon)
        elif not can_see_all(h, buildings):
            max_h_not_all_visible = max(max_h_not_all_visible, h)

    low = 0.0
    high = 1e10  # A sufficiently large number
    ans = 0.0

    for _ in range(100): # Binary search iterations
        mid = (low + high) / 2
        if not can_see_all(mid, buildings):
            ans = mid
            low = mid
        else:
            high = mid

    print(f"{ans:.20f}")

solve()