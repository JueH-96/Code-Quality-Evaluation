def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Allowed replacement letters
    allowed = set("atcoder")

    # Count characters in S and T
    from collections import Counter
    cntS = Counter(S)
    cntT = Counter(T)

    # 1) For any letter not '@' and not in allowed, counts must match exactly
    for ch in cntS:
        if ch != '@' and ch not in allowed:
            if cntS[ch] != cntT.get(ch, 0):
                print("No")
                return
    for ch in cntT:
        if ch != '@' and ch not in allowed:
            if cntT[ch] != cntS.get(ch, 0):
                print("No")
                return

    # 2) Compute how many replacements S and T each need
    #    diffS = total letters in allowed that S is short of compared to T
    #    diffT = total letters in allowed that T is short of compared to S
    diffS = 0
    diffT = 0
    for ch in allowed:
        s_cnt = cntS.get(ch, 0)
        t_cnt = cntT.get(ch, 0)
        if s_cnt < t_cnt:
            diffS += (t_cnt - s_cnt)
        else:
            diffT += (s_cnt - t_cnt)

    # 3) Check if S has enough '@' to cover diffS, and T has enough '@' to cover diffT
    if cntS.get('@', 0) >= diffS and cntT.get('@', 0) >= diffT:
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()