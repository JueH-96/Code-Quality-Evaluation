import heapq

def solve():
    N, M = map(int, input().split())
    items = [list(map(int, input().split())) for _ in range(N)]
    
    pull_tab_cans = []
    regular_cans = []
    can_openers = []
    
    for T, X in items:
        if T == 0:
            pull_tab_cans.append(X)
        elif T == 1:
            regular_cans.append(X)
        else:
            can_openers.append(X)
    
    pull_tab_cans.sort(reverse=True)
    regular_cans.sort(reverse=True)
    
    total_happiness = sum(pull_tab_cans[:M])
    M -= len(pull_tab_cans[:M])
    
    if M > 0:
        can_openers.sort(reverse=True)
        regular_cans = [-x for x in regular_cans]
        heapq.heapify(regular_cans)
        
        for opener in can_openers:
            if M == 0:
                break
            for _ in range(min(opener, M)):
                if regular_cans:
                    total_happiness -= heapq.heappop(regular_cans)
                    M -= 1
                else:
                    break
    
    print(total_happiness)

solve()