def solve():
    n, m = map(int, input().split())
    testimonies = []
    for _ in range(m):
        testimonies.append(list(map(int, input().split())))

    for i in range(2**n):
        confused = bin(i)[2:].zfill(n)
        
        possible = False
        for j in range(2**n):
            honest_liar = bin(j)[2:].zfill(n)
            valid = True
            for a, b, c in testimonies:
                a -= 1
                b -= 1
                
                speaker_confused = confused[a] == '1'
                speaker_honest = honest_liar[a] == '0'
                
                target_honest = honest_liar[b] == '0'

                
                if speaker_confused:
                    speaker_honest = not speaker_honest
                
                if speaker_honest:
                    if (c == 0 and not target_honest) or (c == 1 and target_honest):
                        valid = False
                        break
                else:
                    if (c == 0 and target_honest) or (c == 1 and not target_honest):
                        valid = False
                        break
            
            if valid:
                possible = True
                break
        
        if possible:
            print(confused)
            return

    print("-1")

solve()