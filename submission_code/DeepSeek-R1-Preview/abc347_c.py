import sys

def main():
    N, A, B = map(int, sys.stdin.readline().split())
    W = A + B
    if W == 0:
        print("No")
        return
    D = list(map(int, sys.stdin.readline().split()))
    
    intervals = []
    for d in D:
        m = d % W
        l = (1 - m) % W
        r = (A - m) % W
        intervals.append((l, r))
    
    curr_start, curr_end = 0, W - 1
    
    for l, r in intervals:
        # Determine if current interval is wrapping
        curr_wrap = curr_start > curr_end
        new_wrap = l > r
        
        # Case handling
        if not curr_wrap and not new_wrap:
            # Both are non-wrapping
            new_s = max(curr_start, l)
            new_e = min(curr_end, r)
            if new_s > new_e:
                print("No")
                return
            curr_start, curr_end = new_s, new_e
        elif not curr_wrap and new_wrap:
            # Current is non-wrapping, new is wrapping
            # Check overlaps with [l, W-1] and [0, r]
            overlap1_s = max(curr_start, l)
            overlap1_e = min(curr_end, W-1)
            valid1 = overlap1_s <= overlap1_e
            overlap2_s = max(curr_start, 0)
            overlap2_e = min(curr_end, r)
            valid2 = overlap2_s <= overlap2_e
            if not valid1 and not valid2:
                print("No")
                return
            # Merge the overlaps
            if valid1 and valid2:
                new_s = min(overlap1_s, overlap2_s)
                new_e = max(overlap1_e, overlap2_e)
            elif valid1:
                new_s, new_e = overlap1_s, overlap1_e
            else:
                new_s, new_e = overlap2_s, overlap2_e
            curr_start, curr_end = new_s, new_e
        elif curr_wrap and not new_wrap:
            # Current is wrapping, new is non-wrapping
            # Check overlaps with [curr_start, W-1] and [0, curr_end]
            overlap1_s = max(curr_start, l)
            overlap1_e = min(W-1, r)
            valid1 = overlap1_s <= overlap1_e
            overlap2_s = max(0, l)
            overlap2_e = min(curr_end, r)
            valid2 = overlap2_s <= overlap2_e
            if not valid1 and not valid2:
                print("No")
                return
            # Merge overlaps
            if valid1 and valid2:
                new_s = min(overlap1_s, overlap2_s)
                new_e = max(overlap1_e, overlap2_e)
            elif valid1:
                new_s, new_e = overlap1_s, overlap1_e
            else:
                new_s, new_e = overlap2_s, overlap2_e
            curr_start, curr_end = new_s, new_e
        else:
            # Both are wrapping
            # Intersection is [max(curr_start, l), W-1] and [0, min(curr_end, r)]
            new_s = max(curr_start, l)
            new_e = min(curr_end, r)
            if new_s > W - 1 and new_e < 0:
                print("No")
                return
            # If new_s <= new_e, it's a non-wrapping interval; else, wrapping
            curr_start, curr_end = new_s, new_e
        
        # Check if current interval is empty
        if curr_start <= curr_end:
            if curr_start > W - 1 or curr_end < 0:
                print("No")
                return
        else:
            if (curr_start > W - 1 and curr_end < 0) or (curr_start == 0 and curr_end == W - 1):
                print("No")
                return
    
    print("Yes")

if __name__ == "__main__":
    main()