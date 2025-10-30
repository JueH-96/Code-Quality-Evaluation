import sys

def calculate_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3):
    def intersection_volume(ax1, ay1, az1, ax2, ay2, az2):
        dx = max(0, min(ax1 + 7, ax2 + 7) - max(ax1, ax2))
        dy = max(0, min(ay1 + 7, ay2 + 7) - max(ay1, ay2))
        dz = max(0, min(az1 + 7, az2 + 7) - max(az1, az2))
        return dx * dy * dz

    v12 = intersection_volume(a1, b1, c1, a2, b2, c2)
    v13 = intersection_volume(a1, b1, c1, a3, b3, c3)
    v23 = intersection_volume(a2, b2, c2, a3, b3, c3)
    v123 = intersection_volume(max(a1, a2), max(b1, b2), max(c1, c2), a3, b3, c3)

    v1 = 7**3 - (v12 + v13 - v123)
    v2 = 7**3 - (v12 + v23 - v123)
    v3 = 7**3 - (v13 + v23 - v123)

    v_1 = v1 + v2 + v3 - 2 * (v12 + v13 + v23) + 3 * v123
    v_2 = v12 + v13 + v23 - 2 * v123
    v_3 = v123

    return v_1, v_2, v_3

def find_solution(V1, V2, V3):
    for a1 in range(-100, 101):
        for b1 in range(-100, 101):
            for c1 in range(-100, 101):
                for a2 in range(-100, 101):
                    for b2 in range(-100, 101):
                        for c2 in range(-100, 101):
                            for a3 in range(-100, 101):
                                for b3 in range(-100, 101):
                                    for c3 in range(-100, 101):
                                        v_1, v_2, v_3 = calculate_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                                        if v_1 == V1 and v_2 == V2 and v_3 == V3:
                                            return a1, b1, c1, a2, b2, c2, a3, b3, c3
    return None

def main():
    input = sys.stdin.read
    data = input().split()

    V1 = int(data[0])
    V2 = int(data[1])
    V3 = int(data[2])

    solution = find_solution(V1, V2, V3)
    if solution:
        print("Yes")
        print(" ".join(map(str, solution)))
    else:
        print("No")

if __name__ == "__main__":
    main()