def solve():
    N = int(input())
    H = list(map(int, input().split()))

    if N == 0: # Technically N >= 1 by constraints, but good to be robust
        print(-1)
        return
    
    if N == 1: # Only one building, so no other building can be taller
        print(-1)
        return

    first_building_height = H[0]
    found_position = -1

    # Iterate from the second building (index 1) to the last building
    for i in range(1, N):
        if H[i] > first_building_height:
            found_position = i + 1 # Convert 0-based index to 1-based position
            break # Found the leftmost, so stop
    
    print(found_position)

if __name__ == '__main__':
    solve()