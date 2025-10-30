import sys

def main():
    N, A, B = map(int, sys.stdin.readline().split())
    D = list(map(int, sys.stdin.readline().split()))
    M = A + B

    L_list = []
    R_list = []

    for d in D:
        di = d % M
        L = (-di) % M
        R = (A - 1 - di) % M
        L_list.append(L)
        R_list.append(R)

    max_L = max(L_list)
    min_R = min(R_list)

    if max_L <= min_R:
        print("Yes")
        return

    # Check possibility a
    possible_a = True
    for i in range(N):
        L = L_list[i]
        R = R_list[i]
        if L <= R:  # normal interval
            if L > min_R:
                possible_a = False
                break
    if possible_a:
        print("Yes")
        return

    # Check possibility b
    possible_b = True
    for i in range(N):
        L = L_list[i]
        R = R_list[i]
        if L <= R:  # normal interval
            if R < max_L:
                possible_b = False
                break
    if possible_b:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()