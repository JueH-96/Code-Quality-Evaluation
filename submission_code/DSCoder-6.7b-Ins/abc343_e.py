def solve(V1, V2, V3):
    for a1 in range(-100, 101):
        for b1 in range(-100, 101):
            for c1 in range(-100, 101):
                for a2 in range(-100, 101):
                    for b2 in range(-100, 101):
                        for c2 in range(-100, 101):
                            for a3 in range(-100, 101):
                                for b3 in range(-100, 101):
                                    for c3 in range(-100, 101):
                                        C1 = (max(0, min(a1+7, 100)-a1+1) * max(0, min(b1+7, 100)-b1+1) * max(0, min(c1+7, 100)-c1+1))
                                        C2 = (max(0, min(a2+7, 100)-a2+1) * max(0, min(b2+7, 100)-b2+1) * max(0, min(c2+7, 100)-c2+1))
                                        C3 = (max(0, min(a3+7, 100)-a3+1) * max(0, min(b3+7, 100)-b3+1) * max(0, min(c3+7, 100)-c3+1))
                                        if C1 == V1 and C2 == V2 and C3 == V3:
                                            return (a1, b1, c1, a2, b2, c2, a3, b3, c3)
    return None

V1, V2, V3 = map(int, input().split())
result = solve(V1, V2, V3)
if result is None:
    print("No")
else:
    print("Yes")
    print(" ".join(map(str, result)))