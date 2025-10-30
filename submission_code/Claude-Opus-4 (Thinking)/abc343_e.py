V1, V2, V3 = map(int, input().split())

# Check necessary condition
if V1 != 1029 - 2*V2 - 3*V3:
    print("No")
else:
    # Helper functions
    def overlap_dim(a1, a2):
        return max(0, min(a1+7, a2+7) - max(a1, a2))
    
    def overlap(a1, b1, c1, a2, b2, c2):
        return overlap_dim(a1, a2) * overlap_dim(b1, b2) * overlap_dim(c1, c2)
    
    def triple_overlap(a1, b1, c1, a2, b2, c2, a3, b3, c3):
        dx = max(0, min(a1+7, a2+7, a3+7) - max(a1, a2, a3))
        dy = max(0, min(b1+7, b2+7, b3+7) - max(b1, b2, b3))
        dz = max(0, min(c1+7, c2+7, c3+7) - max(c1, c2, c3))
        return dx * dy * dz
    
    found = False
    
    # Search with C1 at origin
    for a2 in range(-14, 15):
        if found: break
        for b2 in range(-14, 15):
            if found: break
            for c2 in range(-14, 15):
                if found: break
                for a3 in range(-14, 15):
                    if found: break
                    for b3 in range(-14, 15):
                        if found: break
                        for c3 in range(-14, 15):
                            v12 = overlap(0, 0, 0, a2, b2, c2)
                            v13 = overlap(0, 0, 0, a3, b3, c3)
                            v23 = overlap(a2, b2, c2, a3, b3, c3)
                            v123 = triple_overlap(0, 0, 0, a2, b2, c2, a3, b3, c3)
                            
                            if v123 == V3 and v12 + v13 + v23 - 3*v123 == V2:
                                print("Yes")
                                print(0, 0, 0, a2, b2, c2, a3, b3, c3)
                                found = True
                                break
    
    if not found:
        print("No")