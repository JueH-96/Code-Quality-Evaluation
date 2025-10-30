def solve():
    n = int(input())
    s = input()

    while True:
        original_s = s
        new_s_list = list(s)
        made_deletion = False

        i = 0
        while i < len(new_s_list):
            if new_s_list[i] == '(':
                j = i + 1
                while j < len(new_s_list):
                    if new_s_list[j] == ')':
                        is_removable = True
                        for k in range(i + 1, j):
                            if new_s_list[k] == '(' or new_s_list[k] == ')':
                                is_removable = False
                                break
                        if is_removable:
                            new_s_list = new_s_list[:i] + new_s_list[j+1:]
                            made_deletion = True
                            break
                    j += 1
                if made_deletion:
                    break
            i += 1

        s = "".join(new_s_list)
        if not made_deletion:
            break

    print(s)

solve()