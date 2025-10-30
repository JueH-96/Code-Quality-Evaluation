# from collections import deque
s = str(input())
t = str(input())

ans = deque()
candi = [s]
reverse = 0
while True:
    if reverse == 0:
        if set(candi) & set('abcdefghijklmnopqrstuvwxyz'):
            val = 'a'
        else:
            print(0)
            exit()
    if reverse == 1:
        val = s[-1]

    zipper = zip(s[:-1], t)

    sa = []
    for a, b in list(zipper):
        if a != b:
            s = a + val
            sa.append(val)
            ans.appendleft(s)
            break
    else:
        print(len(ans))
        [print(i) for i in list(ans)]
        break

    a = min(sa)
    s = s[:-1] + a

    if t and s[-1] != t[-1]:
        reverse = 1

    if s == t:
        ans.append(t)
        print(len(ans))
        [print(i) for i in list(ans)]
        break
    #     for i in range(len(s)):
    #         if s[i] == t[i]:
    #             val = s[i]
    #         if s[i] == t[i-1]:
    #             val = t[i-1]
    #         elif s[i-1] != t[i-1]:
    #             val = t[i-1]
    #             break
    #     s = s[:-1] + val
    # else:
    #     for i in range(len(s)):
    #         if s[i] != t[i]:
    #             val = t[i]
    #             break
    #     s = s[:i] + val

    # correct = []
    # if s not in candi:
    #     ans.appendleft(s)
    #     correct.append(s)

    # if t not in candi:
    #     ans.append(t)
    #     correct.append(t)

    # for i in set(candi):
    #     if i != s and i not in correct:
    #         ans.append(i)
    #     if i != t and i not in correct:
    #         ans.append(i)        

    # print(s)