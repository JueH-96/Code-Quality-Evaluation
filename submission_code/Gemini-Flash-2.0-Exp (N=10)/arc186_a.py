def solve():
    n, q = map(int, input().split())
    queries = [int(input()) for _ in range(q)]

    possible_fixed_counts = set()
    
    # Case 1: All elements are fixed (all rows and columns have the same sum)
    possible_fixed_counts.add(n * n)
    
    # Case 2: All elements are not fixed (all rows and columns have different sums)
    if n >= 2:
        possible_fixed_counts.add(0)
    
    # Case 3: Some elements are fixed
    if n >= 2:
        for k in range(1, n):
            possible_fixed_counts.add(n * k)
    
    for query in queries:
        if query in possible_fixed_counts:
            print("Yes")
        else:
            print("No")

solve()