def solve():
    n, m = map(int, input().split())
    testimonies = []
    for _ in range(m):
        testimonies.append(list(map(int, input().split())))

    for i in range(2**n):
        confused = bin(i)[2:].zfill(n)
        
        def check_contradiction(honest_liar):
            for a, b, c in testimonies:
                a -= 1
                b -= 1
                
                a_confused = int(confused[a])
                b_honest = int(honest_liar[b])
                
                a_honest = int(honest_liar[a])
                
                expected_testimony = b_honest
                
                if a_confused == 1:
                    expected_testimony = 1 - expected_testimony
                    
                if (a_honest == 1 and expected_testimony != c) or (a_honest == 0 and expected_testimony == c):
                    return False
            return True

        
        valid = False
        for j in range(2**n):
            honest_liar = bin(j)[2:].zfill(n)
            if check_contradiction(honest_liar):
                valid = True
                break
        
        if valid:
            print(confused)
            return

    print("-1")

solve()