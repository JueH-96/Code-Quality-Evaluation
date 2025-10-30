# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    s = input()
    
    ans = 0
    
    for i in range(1 << n):
        bought = 0
        plain = m
        logo = 0
        worn_plain = 0
        worn_logo = 0
        
        possible = True
        
        for j in range(n):
            if s[j] == '0':
                worn_plain = 0
                worn_logo = 0
            elif s[j] == '1':
                if worn_plain < plain:
                    worn_plain += 1
                elif worn_logo < logo:
                    worn_logo += 1
                elif plain + logo < m + bought:
                    if plain < m:
                        plain += 1
                        worn_plain +=1
                    else:
                        logo += 1
                        worn_logo += 1
                        bought +=1
                
                else:
                    possible = False
                    break
            elif s[j] == '2':
                if worn_logo < logo:
                    worn_logo += 1
                elif plain + logo < m + bought:
                    logo += 1
                    worn_logo += 1
                    bought += 1
                else:
                    possible = False
                    break
        
        if possible:
            if ans == 0 or bought < ans:
                ans = bought
                
    print(ans)

solve()