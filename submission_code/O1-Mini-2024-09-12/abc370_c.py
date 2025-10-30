# YOUR CODE HERE
def main():
    import sys

    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    if S == T:
        print(0)
        return

    X = []
    S_list = list(S)
    T_list = list(T)
    n = len(S_list)

    while S_list != T_list:
        candidates = []
        for i in range(n):
            if S_list[i] != T_list[i]:
                temp = S_list.copy()
                temp[i] = T_list[i]
                candidates.append((''.join(temp), i))
        if not candidates:
            break
        # Find the minimal string
        candidates.sort()
        min_str, pos = candidates[0]
        X.append(min_str)
        S_list[pos] = T_list[pos]

    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    main()