k = int(input())
s = input().strip()
t = input().strip()

if s == t:
    print("Yes")
else:
    len_s = len(s)
    len_t = len(t)
    if abs(len_s - len_t) > 1:
        print("No")
    else:
        if len_s == len_t:
            diff = 0
            for a, b in zip(s, t):
                if a != b:
                    diff += 1
                    if diff > 1:
                        break
            print("Yes" if diff == 1 else "No")
        elif len_t == len_s + 1:
            i = j = 0
            while i < len_s and j < len_t:
                if s[i] == t[j]:
                    i += 1
                    j += 1
                else:
                    j += 1
                    if t[j:] == s[i:]:
                        print("Yes")
                    else:
                        print("No")
                    break
            else:
                print("Yes")
        else:  # len_s == len_t + 1
            i = j = 0
            while i < len_s and j < len_t:
                if s[i] == t[j]:
                    i += 1
                    j += 1
                else:
                    i += 1
                    if s[i:] == t[j:]:
                        print("Yes")
                    else:
                        print("No")
                    break
            else:
                print("Yes")