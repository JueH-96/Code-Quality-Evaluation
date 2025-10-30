def solve():
    n, m = map(int, input().split())
    testimonies = []
    for _ in range(m):
        testimonies.append(list(map(int, input().split())))

    for i in range(2**n):
        confused = bin(i)[2:].zfill(n)
        consistent = True
        for j in range(2**n):
            honest = bin(j)[2:].zfill(n)
            
            valid_testimony = True
            for a, b, c in testimonies:
                a -= 1
                b -= 1
                
                a_confused = confused[a] == '1'
                b_honest = honest[b] == '1'
                
                a_honest = honest[a] == '1'
                
                testimony_truth = (c == 0 and b_honest) or (c == 1 and not b_honest)
                
                
                if a_confused:
                    if a_honest:
                        if not testimony_truth:
                            valid_testimony = False
                            break
                    else:
                        if testimony_truth:
                            valid_testimony = False
                            break
                else:
                    if a_honest:
                        if not testimony_truth:
                            valid_testimony = False
                            break
                    else:
                        if testimony_truth:
                            valid_testimony = False
                            break
            if not valid_testimony:
                consistent = False
                break
        if consistent:
            print(confused)
            return

    print("-1")

solve()