import sys

def main():
    N, A, B = map(int, sys.stdin.readline().split())
    D_list = list(map(int, sys.stdin.readline().split()))
    T = A + B
    forbidden = []
    
    for D in D_list:
        L = (- (D - 1)) % T
        R = L + A
        if R <= T:
            if 0 < L:
                forbidden.append((0, L))
            if R < T:
                forbidden.append((R, T))
        else:
            forbidden_start = R - T
            forbidden.append((forbidden_start, L))
    
    if not forbidden:
        print("Yes")
        return
    
    # Sort the forbidden intervals by start time
    forbidden.sort()
    merged = []
    for start, end in forbidden:
        if not merged:
            merged.append([start, end])
        else:
            last_start, last_end = merged[-1]
            if start > last_end:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
    
    current_end = 0
    for s, e in merged:
        if s > current_end:
            break
        current_end = max(current_end, e)
    
    if current_end == T:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()