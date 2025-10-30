def solve():
    n = input()
    counts = {}
    for digit in n:
        counts[digit] = counts.get(digit, 0) + 1
    
    if counts.get('1', 0) == 1 and counts.get('2', 0) == 2 and counts.get('3', 0) == 3 and len(counts) <= 3:
        print("Yes")
    else:
        print("No")

solve()