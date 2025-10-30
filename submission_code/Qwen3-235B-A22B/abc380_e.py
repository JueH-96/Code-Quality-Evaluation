import sys
from sys import stdin

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    N_Q = input().split()
    if not N_Q:
        return
    N = int(N_Q[0])
    Q = int(N_Q[1])
    
    max_color = N + 2  # To handle N+1 safely
    left = list(range(max_color))  # 0-based index up to N+1 inclusive
    right = list(range(max_color))
    color = list(range(max_color))
    cnt = [0] * (max_color)  # cnt[c] stores the number of cells of color c
    
    for i in range(1, N+1):
        cnt[i] = 1
    
    def find_left(x):
        while left[x] != x:
            # Path compression by halving
            left[x] = left[left[x]]
            x = left[x]
        return x
    
    for _ in range(Q):
        query = input().split()
        if not query:
            continue
        if query[0] == '1':
            x = int(query[1])
            c = int(query[2])
            L = find_left(x)
            old_c = color[L]
            if old_c == c:
                continue  # No change needed
            old_size = right[L] - L + 1
            cnt[old_c] -= old_size
            cnt[c] += old_size
            # Update the color of the current block
            color[L] = c
            
            # Merge with left neighbor
            if L > 1:
                L_left = find_left(L - 1)
                if color[L_left] == c:
                    new_L = min(L, L_left)
                    new_R = max(right[L], right[L_left])
                    # Update parent pointers
                    if L != new_L:
                        left[L] = new_L
                    else:
                        left[L_left] = new_L
                    # Update new_L's data
                    right[new_L] = new_R
                    color[new_L] = c
                    L = new_L  # Point to merged block
            
            # Merge with right neighbor
            current_R = right[L]
            if current_R < N:
                R_plus = current_R + 1
                L_right = find_left(R_plus)
                if color[L_right] == c:
                    new_L = min(L, L_right)
                    new_R = max(right[L], right[L_right])
                    # Update parent pointers
                    if L != new_L:
                        left[L] = new_L
                    else:
                        left[L_right] = new_L
                    # Update new_L's data
                    right[new_L] = new_R
                    color[new_L] = c
                    L = new_L  # Point to merged block
                    current_R = right[L]  # Update current_R after merge
                    
        elif query[0] == '2':
            c = int(query[1])
            print(cnt[c])
            
if __name__ == "__main__":
    main()