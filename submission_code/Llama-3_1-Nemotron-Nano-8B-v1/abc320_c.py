M = int(input())
S = [input().strip() for _ in range(3)]

min_d = float('inf')

for c in '0123456789':
    # Collect residues for each reel
    res1 = [i for i, ch in enumerate(S[0]) if ch == c]
    res2 = [i for i, ch in enumerate(S[1]) if ch == c]
    res3 = [i for i, ch in enumerate(S[2]) if ch == c]
    
    if not (res1 and res2 and res3):
        continue  # Skip if any reel lacks this character
    
    # Check up to 3*M as a reasonable upper bound
    max_d = 3 * M
    found = False
    for D in range(max_d + 1):
        # Generate possible t values for each reel up to D
        possible1 = []
        for r in res1:
            k = 0
            while True:
                t = r + k * M
                if t > D:
                    break
                possible1.append(t)
                k += 1
        possible2 = []
        for r in res2:
            k = 0
            while True:
                t = r + k * M
                if t > D:
                    break
                possible2.append(t)
                k += 1
        possible3 = []
        for r in res3:
            k = 0
            while True:
                t = r + k * M
                if t > D:
                    break
                possible3.append(t)
                k += 1
        
        # Check if any combination of t1, t2, t3 are distinct
        if not (possible1 and possible2 and possible3):
            continue
        
        # Brute-force check all combinations
        valid = False
        for t1 in possible1:
            for t2 in possible2:
                if t1 == t2:
                    continue
                for t3 in possible3:
                    if t3 != t1 and t3 != t2:
                        valid = True
                        break
                if valid:
                    break
            if valid:
                break
        if valid:
            if D < min_d:
                min_d = D
            found = True
            break  # No need to check larger D for this c
    
if min_d != float('inf'):
    print(min_d)
else:
    print(-1)