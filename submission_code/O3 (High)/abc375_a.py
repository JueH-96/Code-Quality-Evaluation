import sys

def main() -> None:
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    N = int(data[0].strip())
    S = data[1].strip() if len(data) > 1 else ""
    
    # The pattern requires three consecutive seats, so if N < 3,
    # it is impossible for any index to satisfy the condition.
    if N < 3:
        print(0)
        return
    
    ans = 0
    # Iterate over valid starting indices (0-based) up to N-3 inclusive.
    for i in range(N - 2):
        if S[i] == '#' and S[i + 1] == '.' and S[i + 2] == '#':
            ans += 1
    
    print(ans)

if __name__ == "__main__":
    main()