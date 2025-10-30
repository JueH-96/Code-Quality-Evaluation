# YOUR CODE HERE
x, y, z = map(int, input().split())
s = input()
ans = 0
caps_lock = False
for i in range(len(s)):
    if s[i] == 'a':
        if caps_lock:
            ans += y
        else:
            ans += x
    else:
        if caps_lock:
            ans += x
        else:
            ans += y
if s.count('a') == len(s) or s.count('A') == len(s):
    print(ans)
else:
    ans1 = ans
    ans2 = ans + z
    caps_lock = True
    ans3 = z
    for i in range(len(s)):
        if s[i] == 'a':
            if caps_lock:
                ans3 += x
            else:
                ans3 += y
        else:
            if caps_lock:
                ans3 += y
            else:
                ans3 += x
    print(min(ans1, ans2, ans3))