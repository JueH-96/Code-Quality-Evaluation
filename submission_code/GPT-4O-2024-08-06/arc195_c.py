def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        R = int(data[index])
        B = int(data[index + 1])
        index += 2
        
        if R > 0 and B > 0:
            # Mixed colors, always possible
            results.append("Yes")
            for i in range(R + B):
                if i % 2 == 0:
                    if R > 0:
                        results.append(f"R {1 + i // 2} {1}")
                        R -= 1
                    else:
                        results.append(f"B {1 + i // 2} {2}")
                        B -= 1
                else:
                    if B > 0:
                        results.append(f"B {1 + i // 2} {2}")
                        B -= 1
                    else:
                        results.append(f"R {1 + i // 2} {1}")
                        R -= 1
        elif R > 0:
            # All red pieces
            if R >= 4:
                results.append("Yes")
                results.append("R 1 1")
                results.append("R 1 2")
                results.append("R 2 2")
                results.append("R 2 1")
                for i in range(4, R):
                    results.append(f"R {1 + (i - 4) // 2} {3 + (i - 4) % 2}")
            else:
                results.append("No")
        elif B > 0:
            # All blue pieces
            if B >= 2:
                results.append("Yes")
                results.append("B 1 1")
                results.append("B 2 2")
                for i in range(2, B):
                    results.append(f"B {1 + i} {1 + i}")
            else:
                results.append("No")
    
    sys.stdout.write("
".join(results) + "
")