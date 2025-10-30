def solve():
    a, b = map(int, input().split())
    
    possible_x = set()

    # Case 1: A, B, x
    x = 2 * b - a
    possible_x.add(x)
    
    # Case 2: A, x, B
    x = (a + b) / 2
    if x == int(x):
        possible_x.add(int(x))
    
    # Case 3: x, A, B
    x = 2 * a - b
    possible_x.add(x)
    
    # Case 4: B, A, x
    x = 2 * a - b
    possible_x.add(x)
    
    # Case 5: B, x, A
    x = (a + b) / 2
    if x == int(x):
        possible_x.add(int(x))
    
    # Case 6: x, B, A
    x = 2 * b - a
    possible_x.add(x)
    
    print(len(possible_x))

solve()