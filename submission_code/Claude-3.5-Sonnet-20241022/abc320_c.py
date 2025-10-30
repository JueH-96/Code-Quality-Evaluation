M = int(input())
S1 = input()
S2 = input()
S3 = input()

def find_min_time():
    # Try each possible final time t from 0 to M*3
    for t in range(M*3):
        # For each time t, try all possible combinations of when to stop each reel
        # such that total time <= t
        for t1 in range(t+1):
            c1 = S1[t1 % M]
            for t2 in range(t+1):
                c2 = S2[t2 % M]
                if c2 != c1:
                    continue
                for t3 in range(t+1):
                    c3 = S3[t3 % M]
                    if c3 != c1:
                        continue
                    
                    # Found a valid solution where all characters match
                    # and all reels are stopped by time t
                    if max(t1, t2, t3) == t:
                        return t
    return -1

print(find_min_time())