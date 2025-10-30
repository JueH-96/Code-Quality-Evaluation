def can_see_from(buildings, start, end):
    max_height = buildings[start]
    for i in range(start + 1, end):
        if buildings[i] >= max_height:
            return False
        max_height = max(max_height, buildings[i])
    return True

def count_visible_buildings(buildings, l, r):
    visible_count = 0
    max_height_l = buildings[l]
    max_height_r = buildings[r]
    
    for j in range(r + 1, len(buildings)):
        if buildings[j] > max_height_l and buildings[j] > max_height_r:
            if can_see_from(buildings, l, j) and can_see_from(buildings, r, j):
                visible_count += 1
    return visible_count

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    first_line = data[0].split()
    N = int(first_line[0])
    Q = int(first_line[1])
    
    heights = list(map(int, data[1].split()))
    
    results = []
    for i in range(2, 2 + Q):
        l, r = map(int, data[i].split())
        # Convert to 0-based index
        l -= 1
        r -= 1
        result = count_visible_buildings(heights, l, r)
        results.append(result)
    
    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()