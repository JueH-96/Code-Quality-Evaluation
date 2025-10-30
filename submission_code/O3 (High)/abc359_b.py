import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    first_pos = [-1] * (N + 1)   # 1-indexed, store first occurrence
    ans = 0
    
    for idx, color in enumerate(A, start=1):  # positions are 1-indexed
        if first_pos[color] == -1:
            first_pos[color] = idx           # remember first occurrence
        else:
            # second occurrence: check the distance
            if idx - first_pos[color] == 2:
                ans += 1
    
    print(ans)

if __name__ == "__main__":
    main()