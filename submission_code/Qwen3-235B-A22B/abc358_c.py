import sys
import itertools

def main():
    N, M = map(int, sys.stdin.readline().split())
    masks = []
    for _ in range(N):
        s = sys.stdin.readline().strip()
        mask = 0
        for j in range(M):
            if s[j] == 'o':
                mask |= 1 << j
        masks.append(mask)
    
    full_mask = (1 << M) - 1  # bitmask covering all flavors
    
    for k in range(1, N + 1):
        for combo in itertools.combinations(range(N), k):
            combined = 0
            for idx in combo:
                combined |= masks[idx]
            if combined == full_mask:
                print(k)
                return

if __name__ == "__main__":
    main()