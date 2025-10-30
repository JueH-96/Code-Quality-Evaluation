def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    # The letters that '@' can become
    allowed = set("atcoder")
    # Count letters
    from collections import Counter
    cntS = Counter(S)
    cntT = Counter(T)
    # Quick check: for letters not in allowed, counts must match exactly
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
    # Compute how many replacements each side needs
    need_from_S_at = 0  # how many extra letters T needs to cover from S's '@'
    need_from_T_at = 0  # how many extra letters S needs to cover from T's '@'
    for ch in allowed:
        s_cnt = cntS.get(ch, 0)
        t_cnt = cntT.get(ch, 0)
        if s_cnt > t_cnt:
            # T needs (s_cnt - t_cnt) '@' -> replaced to ch
            need_from_T_at += (s_cnt - t_cnt)
        else:
            # S needs (t_cnt - s_cnt) '@'
            need_from_S_at += (t_cnt - s_cnt)
    # Check available '@'
    if need_from_S_at <= cntS.get('@', 0) and need_from_T_at <= cntT.get('@', 0):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()