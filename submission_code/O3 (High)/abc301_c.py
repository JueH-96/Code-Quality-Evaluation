import sys

def main() -> None:
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # frequency tables for a–z
    cnt_s = [0] * 26
    cnt_t = [0] * 26
    at_s = at_t = 0               # number of '@' in each string

    for ch in S:
        if ch == '@':
            at_s += 1
        else:
            cnt_s[ord(ch) - 97] += 1

    for ch in T:
        if ch == '@':
            at_t += 1
        else:
            cnt_t[ord(ch) - 97] += 1

    # letters that an '@' can turn into
    allowed = {ord(c) - 97 for c in "atcoder"}

    # 1. letters outside the allowed set must already match
    for i in range(26):
        if i not in allowed and cnt_s[i] != cnt_t[i]:
            print("No")
            return

    # 2. compute how many of each side's '@' are needed
    need_s = need_t = 0
    for i in allowed:
        if cnt_s[i] < cnt_t[i]:
            need_s += cnt_t[i] - cnt_s[i]     # S has to create these letters
        elif cnt_s[i] > cnt_t[i]:
            need_t += cnt_s[i] - cnt_t[i]     # T has to create these letters

    # 3. check if each side owns enough '@' cards
    if need_s <= at_s and need_t <= at_t:
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()