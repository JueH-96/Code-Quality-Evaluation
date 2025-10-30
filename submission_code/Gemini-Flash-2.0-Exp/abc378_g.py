def solve():
    a, b, m = map(int, input().split())
    ab = a * b
    
    # Precompute the Robinson-Schensted-Knuth correspondence
    young_tableau = [[0] * ab for _ in range(ab)]
    young_tableau[0][0] = 1
    
    for i in range(1, ab - 1):
        new_tableau = [[0] * ab for _ in range(ab)]
        for row in range(ab):
            for col in range(ab):
                if young_tableau[row][col] > 0:
                    # Try adding i+1 to the end of the row
                    new_tableau[row][col] = (new_tableau[row][col] + young_tableau[row][col]) % m
                    
                    # Try bumping the first element in the row that is greater than i+1
                    bumped = i + 1
                    r = row
                    c = col
                    while True:
                        found = False
                        for j in range(c, ab):
                            if young_tableau[r][j] > 0 and young_tableau[r][j] > bumped:
                                temp = young_tableau[r][j]
                                young_tableau[r][j] = bumped
                                bumped = temp
                                c = j + 1
                                found = True
                                break
                        if not found:
                            r += 1
                            if r >= ab:
                                break
                            c = 0
                            young_tableau[r][0] = bumped
                            break
        
        for row in range(ab):
            for col in range(ab):
                young_tableau[row][col] = new_tableau[row][col]
    
    ans = 0
    for row in range(ab):
        for col in range(ab):
            if row + 1 == a and col + 1 == b:
                ans = (ans + young_tableau[row][col]) % m
    
    print(ans)

solve()