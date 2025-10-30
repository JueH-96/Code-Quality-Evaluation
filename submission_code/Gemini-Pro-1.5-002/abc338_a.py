# YOUR CODE HERE
s = input()
if len(s) >= 1 and len(s) <= 100:
    if len(s) == 1:
        if 'A' <= s[0] <= 'Z':
            print("Yes")
        else:
            print("No")
    else:
        if 'A' <= s[0] <= 'Z':
            flag = True
            for i in range(1, len(s)):
                if not ('a' <= s[i] <= 'z'):
                    flag = False
                    break
            if flag:
                print("Yes")
            else:
                print("No")

        else:
            print("No")