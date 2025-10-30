# YOUR CODE HERE
V1, V2, V3 = map(int, input().split())

def solve():
    for a1 in range(-100, 101):
        for b1 in range(-100, 101):
            for c1 in range(-100, 101):
                for a2 in range(-100, 101):
                    for b2 in range(-100, 101):
                        for c2 in range(-100, 101):
                            for a3 in range(-100, 101):
                                for b3 in range(-100, 101):
                                    for c3 in range(-100, 101):
                                        
                                        c1_coords = [(x,y,z) for x in range(a1, a1+7) for y in range(b1, b1+7) for z in range(c1, c1+7)]
                                        c2_coords = [(x,y,z) for x in range(a2, a2+7) for y in range(b2, b2+7) for z in range(c2, c2+7)]
                                        c3_coords = [(x,y,z) for x in range(a3, a3+7) for y in range(b3, b3+7) for z in range(c3, c3+7)]

                                        v1_count = 0
                                        v2_count = 0
                                        v3_count = 0

                                        all_coords = set(c1_coords + c2_coords + c3_coords)

                                        for coord in all_coords:
                                            count = 0
                                            if coord in c1_coords:
                                                count +=1
                                            if coord in c2_coords:
                                                count +=1
                                            if coord in c3_coords:
                                                count +=1
                                            if count == 1:
                                                v1_count += 1
                                            elif count == 2:
                                                v2_count += 1
                                            elif count == 3:
                                                v3_count += 1

                                        if v1_count == V1 and v2_count == V2 and v3_count == V3:
                                            print("Yes")
                                            print(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                                            return

    print("No")

solve()