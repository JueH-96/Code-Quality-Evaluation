s = input()
n = len(s)
if n == 0:
    print("Yes")
else:
    i = 0
    while i < n and s[i] == 'A':
        i += 1
    j = i
    while j < n and s[j] == 'B':
        j += 1
    k = j
    while k < n and s[k] == 'C':
        k += 1
    if k == n and (i > 0 or j > i) and (j > i or k > j) and (k > j or i > 0):
        if i == 0 and j == i and k == j:
            print("Yes")
        elif i > 0 and j > i and k > j:
            print("Yes")
        elif i > 0 and j == i and k > j:
            print("Yes")
        elif i == 0 and j > i and k > j:
            print("Yes")
        elif i > 0 and j > i and k == j:
            print("Yes")
        elif i == 0 and j == i and k > j:
            print("Yes")
        elif i == 0 and j > i and k == j:
            print("Yes")
        elif i > 0 and j == i and k == j:
            print("Yes")
        else:
            print("No")
    else:
        print("No")