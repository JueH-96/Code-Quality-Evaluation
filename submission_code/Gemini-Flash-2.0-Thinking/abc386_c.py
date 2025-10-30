def solve():
    k = int(input())
    s = input()
    t = input()

    if s == t:
        print("Yes")
        return

    if k >= 1:
        # Check for insertion
        if len(t) == len(s) + 1:
            for i in range(len(t)):
                temp_t = list(t)
                char_to_remove = temp_t.pop(i)
                if "".join(temp_t) == s:
                    print("Yes")
                    return

        # Check for deletion
        if len(s) == len(t) + 1:
            for i in range(len(s)):
                temp_s = list(s)
                char_to_remove = temp_s.pop(i)
                if "".join(temp_s) == t:
                    print("Yes")
                    return

        # Check for replacement
        if len(s) == len(t):
            diff_count = 0
            for i in range(len(s)):
                if s[i] != t[i]:
                    diff_count += 1
            if diff_count == 1:
                print("Yes")
                return

    print("No")

solve()