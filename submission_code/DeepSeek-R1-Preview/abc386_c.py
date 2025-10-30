k = int(input())
s = input().strip()
t = input().strip()

len_s = len(s)
len_t = len(t)

if abs(len_s - len_t) > 1:
    print("No")
else:
    if len_s == len_t:
        diff = 0
        for sc, tc in zip(s, t):
            if sc != tc:
                diff += 1
                if diff > 1:
                    break
        print("Yes" if diff <= 1 else "No")
    else:
        if len_s > len_t:
            longer = s
            shorter = t
        else:
            longer = t
            shorter = s
        i = j = 0
        diff = 0
        len_long = len(longer)
        len_short = len(shorter)
        while i < len_long and j < len_short:
            if longer[i] == shorter[j]:
                i += 1
                j += 1
            else:
                i += 1
                diff += 1
                if diff > 1:
                    break
        diff += len_long - i
        print("Yes" if diff <= 1 else "No")