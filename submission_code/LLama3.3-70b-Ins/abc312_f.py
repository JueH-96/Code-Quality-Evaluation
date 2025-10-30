import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    items = []
    for _ in range(N):
        T, X = map(int, sys.stdin.readline().split())
        items.append((T, X))

    # Sort items by type and value
    items.sort(key=lambda x: (x[0], -x[1]))

    # Initialize variables
    pull_tab_cans = []
    regular_cans = []
    can_openers = []
    total_happiness = 0

    # Separate items into categories
    for item in items:
        if item[0] == 0:
            pull_tab_cans.append(item[1])
        elif item[0] == 1:
            regular_cans.append(item[1])
        else:
            can_openers.append(item[1])

    # Calculate happiness from pull-tab cans
    pull_tab_cans.sort(reverse=True)
    total_happiness += sum(pull_tab_cans[:M])

    # Calculate happiness from regular cans and can openers
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)
    i = 0
    j = 0
    while i < len(regular_cans) and j < len(can_openers):
        if M > 0 and can_openers[j] > 0:
            total_happiness += regular_cans[i]
            can_openers[j] -= 1
            M -= 1
            i += 1
        else:
            j += 1

    print(total_happiness)

if __name__ == "__main__":
    solve()