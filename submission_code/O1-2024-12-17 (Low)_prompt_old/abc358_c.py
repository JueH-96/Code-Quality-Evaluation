def solve():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    stands = []
    
    # Read the stands data and convert each stand's flavors into a bitmask.
    # If the j-th character is 'o', set the j-th bit.
    for _ in range(N):
        s = input().strip()
        bitmask = 0
        for j, ch in enumerate(s):
            if ch == 'o':
                bitmask |= (1 << j)
        stands.append(bitmask)

    # We want to buy all flavors, so the "all flavors bitmask" is (1 << M) - 1
    all_flavors = (1 << M) - 1
    answer = N  # worst case, we might need all stands

    # Enumerate all subsets of stands using a bitmask from 1 to (1 << N) - 1
    for subset in range(1, 1 << N):
        flavor_cover = 0
        stand_count = 0
        for i in range(N):
            if subset & (1 << i):
                flavor_cover |= stands[i]
                stand_count += 1
        
        if flavor_cover == all_flavors:
            answer = min(answer, stand_count)

    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()