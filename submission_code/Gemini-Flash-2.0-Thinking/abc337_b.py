def is_extended_a(s):
    return all(c == 'A' for c in s)

def is_extended_b(s):
    return all(c == 'B' for c in s)

def is_extended_c(s):
    return all(c == 'C' for c in s)

s = input()
n = len(s)
is_extended_abc = False

for len_a in range(n + 1):
    for len_b in range(n - len_a + 1):
        len_c = n - len_a - len_b
        if len_c >= 0:
            s_a = s[:len_a]
            s_b = s[len_a:len_a + len_b]
            s_c = s[len_a + len_b:]
            if is_extended_a(s_a) and is_extended_b(s_b) and is_extended_c(s_c):
                is_extended_abc = True
                break
    if is_extended_abc:
        break

if is_extended_abc:
    print("Yes")
else:
    print("No")