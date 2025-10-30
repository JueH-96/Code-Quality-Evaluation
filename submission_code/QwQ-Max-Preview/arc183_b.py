import sys
import bisect

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, K = int(input[ptr]), int(input[ptr+1])
        ptr +=2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        # Check if all elements in B are present in A
        set_A = set(A)
        set_B = set(B)
        if not set_B.issubset(set_A):
            print("No")
            continue
        
        # Preprocess: for each value in A, store sorted list of indices
        from collections import defaultdict
        pos_map = defaultdict(list)
        for idx, val in enumerate(A):
            pos_map[val].append(idx)
        
        possible = True
        for i in range(N):
            target_val = B[i]
            positions = pos_map.get(target_val, [])
            if not positions:
                possible = False
                break
            # Find if there is any position in positions that is within [i-K, i+K]
            left = i - K
            right = i + K
            # Find the first position >= left
            l = bisect.bisect_left(positions, left)
            if l < len(positions) and positions[l] <= right:
                continue
            else:
                possible = False
                break
        
        print("Yes" if possible else "No")

if __name__ == "__main__":
    main()