import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    xh = []
    s = []
    idx = 1
    for _ in range(n):
        x = int(data[idx])
        h = int(data[idx+1])
        xh.append((x, h))
        s.append(h / x)
        idx += 2

    # Check if all buildings are visible at h=0
    visible_at_0 = True
    max_s = -float('inf')
    for i in range(n):
        if s[i] > max_s + 1e-12:
            max_s = s[i]
        else:
            visible_at_0 = False
            break
    if visible_at_0:
        print(-1)
        return

    # Compute h_min using convex hull trick optimized approach
    import bisect

    # Maintain a list of indices in the convex hull
    hull = []
    h_min = 0.0

    for i in range(n):
        x_i, H_i = xh[i]
        s_i = s[i]
        q_i = 1.0 / x_i

        # Binary search to find the best j in hull
        current_max = -float('inf')
        l = 0
        r = len(hull) - 1
        best_val = -float('inf')
        best_idx = -1

        while l <= r:
            mid = (l + r) // 2
            j = hull[mid]
            x_j, H_j = xh[j]
            s_j = s[j]
            q_j = 1.0 / x_j

            # Calculate slope between (q_j, s_j) and (q_i, s_i)
            if q_j == q_i:
                continue  # should not happen
            slope = (s_j - s_i) / (q_j - q_i)

            if slope > best_val:
                best_val = slope
                best_idx = mid

            if mid == 0:
                l = mid + 1
            elif mid == len(hull) - 1:
                r = mid - 1
            else:
                j_prev = hull[mid - 1]
                j_next = hull[mid + 1]
                s_prev = s[j_prev]
                s_next = s[j_next]
                q_prev = 1.0 / xh[j_prev][0]
                q_next = 1.0 / xh[j_next][0]

                slope_prev = (s_prev - s_i) / (q_prev - q_i)
                slope_next = (s_next - s_i) / (q_next - q_i)

                if slope_next > slope:
                    l = mid + 1
                elif slope_prev > slope:
                    r = mid - 1
                else:
                    break

            if best_val > current_max:
                current_max = best_val

        # Check hull edges
        for mid in [0, len(hull) - 1]:
            if mid < 0 or mid >= len(hull):
                continue
            j = hull[mid]
            x_j, H_j = xh[j]
            q_j = 1.0 / x_j
            s_j = s[j]
            if q_j == q_i:
                continue
            slope = (s_j - s_i) / (q_j - q_i)
            if slope > current_max:
                current_max = slope

        if hull:
            if current_max == -float('inf'):
                current_max = -float('inf')
        else:
            current_max = 0.0

        if current_max < 0:
            current_max = 0.0

        if current_max > h_min:
            h_min = current_max

        # Maintain convex hull using cross product
        while len(hull) >= 2:
            j1 = hull[-2]
            j2 = hull[-1]
            x1, H1 = xh[j1]
            x2, H2 = xh[j2]
            q1 = 1.0 / x1
            q2 = 1.0 / x2
            s1 = s[j1]
            s2 = s[j2]

            # Compute cross product ((q2 - q1)*(s_i - s1) - (q_i - q1)*(s2 - s1))
            cross = (q2 - q1) * (s_i - s1) - (q_i - q1) * (s2 - s1)
            if cross >= 0:
                break
            hull.pop()
        hull.append(i)

    print("{0:.20f}".format(h_min))

if __name__ == "__main__":
    main()