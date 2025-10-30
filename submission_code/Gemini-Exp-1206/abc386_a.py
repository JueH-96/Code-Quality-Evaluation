def solve():
    a, b, c, d = map(int, input().split())
    nums = [a, b, c, d]
    counts = {}
    for num in nums:
        counts[num] = counts.get(num, 0) + 1

    if len(counts) == 2:
        if 3 in counts.values() and 1 in counts.values():
            print("Yes")
            return
        if 2 in counts.values() and 2 in counts.values():
            print("Yes")
            return
    elif len(counts) == 3:
        if 2 in counts.values() and 1 in counts.values():
            print("Yes")
            return
    elif len(counts) == 1:
        if 4 in counts.values():
            print("No")
            return
    
    print("No")

solve()