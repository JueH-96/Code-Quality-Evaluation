def solve():
    n, m = map(int, input().split())
    testimonies = []
    for _ in range(m):
        testimonies.append(list(map(int, input().split())))

    for i in range(2**n):
        confused = bin(i)[2:].zfill(n)
        
        valid_assignment_found = False
        for j in range(2**n):
            honest_liar = bin(j)[2:].zfill(n)
            
            consistent = True
            for a, b, c in testimonies:
                a -= 1
                b -= 1
                
                a_confused = int(confused[a])
                a_honest = int(honest_liar[a])
                b_honest = int(honest_liar[b])
                
                if a_confused == 0: # not confused
                    if a_honest == 0: # honest
                        if (b_honest == 0 and c == 1) or (b_honest == 1 and c == 0):
                            consistent = False
                            break
                    else: # liar
                        if (b_honest == 0 and c == 0) or (b_honest == 1 and c == 1):
                            consistent = False
                            break
                else: # confused
                    if a_honest == 0: # honest
                        if (b_honest == 0 and c == 0) or (b_honest == 1 and c == 1):
                            consistent = False
                            break
                    else: # liar
                        if (b_honest == 0 and c == 1) or (b_honest == 1 and c == 0):
                            consistent = False
                            break
            if consistent:
                valid_assignment_found = True
                break
        
        if valid_assignment_found:
            print(confused)
            return
    
    print("-1")

solve()