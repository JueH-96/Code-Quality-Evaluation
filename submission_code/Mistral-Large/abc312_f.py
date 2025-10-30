import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0

    N = int(data[index])
    M = int(data[index + 1])
    index += 2

    items = []
    for _ in range(N):
        T = int(data[index])
        X = int(data[index + 1])
        index += 2
        items.append((T, X))

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
    can_openers.sort(reverse=True)

    max_happiness = 0

    selected_pull_tab_cans = 0
    selected_regular_cans = 0
    selected_can_openers = 0

    while selected_pull_tab_cans + selected_regular_cans + selected_can_openers < M:
        if pull_tab_cans and (selected_pull_tab_cans < len(pull_tab_cans)):
            max_happiness += pull_tab_cans[selected_pull_tab_cans]
            selected_pull_tab_cans += 1
        elif can_openers and (selected_can_openers < len(can_openers)):
            can_opener_uses = can_openers[selected_can_openers]
            selected_can_openers += 1
            while can_opener_uses > 0 and selected_regular_cans < len(regular_cans):
                max_happiness += regular_cans[selected_regular_cans]
                selected_regular_cans += 1
                can_opener_uses -= 1
        else:
            break

    print(max_happiness)

if __name__ == "__main__":
    main()