# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    superior = []
    for _ in range(m):
        superior.append(list(map(int, input().split())))

    strongest_candidates = []
    for i in range(1, n + 1):
        is_strongest = True
        for j in range(1, n + 1):
            if i == j:
                continue
            
            is_i_stronger_than_j = False
            
            
            def check_stronger(person1, person2, current_superior):
                if [person1, person2] in current_superior:
                    return True
                
                for k in range(1, n+1):
                    if k == person1 or k == person2:
                        continue
                    if [person1, k] in current_superior and check_stronger(k, person2, current_superior):
                        return True
                return False

            is_i_stronger_than_j = check_stronger(i,j, superior)
            
            if not is_i_stronger_than_j:
                
                
                
                temp_superior = superior[:]
                temp_superior.append([j,i])
                
                is_consistent = True
                for p1 in range(1, n+1):
                    for p2 in range(1, n+1):
                        if p1 == p2:
                            continue
                        
                        is_p1_stronger_p2 = check_stronger(p1, p2, temp_superior)
                        is_p2_stronger_p1 = check_stronger(p2, p1, temp_superior)
                        
                        if is_p1_stronger_p2 and is_p2_stronger_p1:
                            is_consistent = False
                            break
                        if not is_p1_stronger_p2 and not is_p2_stronger_p1:
                            is_consistent = False
                            break
                    if not is_consistent:
                        break
                
                if is_consistent:
                    is_strongest = False
                    break
        
        if is_strongest:
            strongest_candidates.append(i)

    if len(strongest_candidates) == 1:
        print(strongest_candidates[0])
    else:
        print(-1)

solve()